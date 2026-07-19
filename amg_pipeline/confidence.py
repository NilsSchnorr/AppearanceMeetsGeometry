"""
run_confidence(base_config, ...) — R3 #7: per-pixel assignment confidence of
the softmax ensemble.

The segmentation ensemble (ensemble.run_ensemble) averages the per-window
softmax outputs of the N source-run checkpoints, merges the windows
center-weighted, and then DISCARDS the merged probabilities after the argmax.
This module re-runs the identical inference but keeps the merged per-pixel
class probabilities, from which it derives:

  - a top-1 confidence raster per wall (uint16 PNG, 0..65535 == 0..1) —
    input for the R3 #7 heatmap figure;
  - for one wall (probs_wall, default "wall1") the full per-class probability
    stack (float16 .npz, key "probs", HWC) — input for the pixel-readout
    figure ("behind this pixel: xx % ashlar, xx % polygonal, ...");
  - confidence statistics on two bases, each per wall: (a) within the same
    GT-derived ROI as every other paper metric (config.roi_operation /
    config.kernel_radius) and (b) over GT stone pixels only (gt > 0),
    matching the drafted "confidence for stone pixels" wording — mean and
    median top-1 confidence plus the fraction of pixels above 0.9; a
    pixel-pooled AllWalls row is added only when more than one wall is run;
  - a sanity column `raster_agreement`: the fraction of pixels (full image)
    where the argmax of the recomputed probabilities equals the saved
    ensemble raster. Expect ~1.0; anything clearly below means this pass did
    NOT reproduce the ensemble (wrong checkpoints or config) and the
    statistics must not be used.

Mechanics mirror ensemble.run_ensemble exactly: same checkpoints, same
windowing, same center-weighted merge (the accumulation below must be kept IN
SYNC with segment.combine_center_weighted; it streams per window instead of
collecting a windows_data list, which is numerically identical and lighter on
memory). The source experiment is only ever read. Outputs land under the out
experiment:

    <experiments_root>/<out_experiment>/confidence/<run_id>/
        <wall>_confidence.png       (uint16, scaled 0..65535)
        <wall>_probs.npz            (probs_wall only)
        confidence_summary.csv      (per wall + pooled AllWalls row)

Skip-if-exists: if a wall's confidence PNG is present (and force=False) the
inference is skipped and the statistics are recomputed from the PNG; the
agreement check is then NaN for that wall (no probabilities were recomputed).
"""

import dataclasses
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from . import paths
from .config import make_run_id
from .data import rgb_to_class_mask
from .evaluate import generate_roi
from .segment import (_build_wall_stack, _load_model, _segment_window,
                      create_sliding_windows)

Image.MAX_IMAGE_PIXELS = None

N_HIST_BINS = 10000  # resolution of the pooled AllWalls median
CONF_THRESHOLD = 0.9


def _center_weight(window_size):
    """The window weighting of segment.combine_center_weighted — keep in sync."""
    h, w = window_size
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    return 1.0 - (dist / max_dist) * 0.7


def _confidence_dir(config):
    return os.path.join(paths.experiment_dir(config), "confidence",
                        make_run_id(config))


def _wall_probabilities(models, cfg, wall, device):
    """Full-wall merged class probabilities (HWC float32), streaming the
    center-weighted accumulation window by window."""
    stack = _build_wall_stack(cfg, wall)
    windows, positions = create_sliding_windows(stack, cfg.window_size,
                                                cfg.stride)
    height, width = stack.shape[:2]
    win_w = _center_weight(cfg.window_size)
    class_scores = None
    weights = np.zeros((height, width), dtype=np.float32)

    for i, (win, (y0, x0, y1, x1)) in enumerate(
            zip(tqdm(windows, desc=f"{wall}: windows x{len(models)} models"),
                positions)):
        probs_sum = None
        for m in models:
            _, probs = _segment_window(m, win, cfg.model_size,
                                       cfg.window_size, device)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        probs_mean = probs_sum / float(len(models))
        if class_scores is None:
            n_classes = probs_mean.shape[0]
            class_scores = np.zeros((height, width, n_classes),
                                    dtype=np.float32)
        ah, aw = y1 - y0, x1 - x0
        for c in range(probs_mean.shape[0]):
            class_scores[y0:y1, x0:x1, c] += probs_mean[c, :ah, :aw] * win_w[:ah, :aw]
        weights[y0:y1, x0:x1] += win_w[:ah, :aw]
        if i % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    weights = np.maximum(weights, 1e-6)
    class_scores /= np.expand_dims(weights, axis=2)
    return class_scores


def run_confidence(base_config, source_experiment, out_experiment,
                   channel_variants=(7,), n_runs=5,
                   checkpoint_filename="model.pth", probs_wall="wall1",
                   force=False):
    """
    For each variant: load the N run checkpoints of `source_experiment`,
    recompute the mean-softmax ensemble probabilities for every test wall,
    save confidence rasters + statistics under `out_experiment`, and verify
    agreement with the saved ensemble segmentation.

    Match the parameters to the segmentation ensemble's manifest provenance
    columns (source_experiment, n_models, checkpoint_filename) — otherwise the
    agreement check will fail and the statistics are meaningless.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ch in channel_variants:
        out_cfg = dataclasses.replace(base_config, channels=ch, run_number=1,
                                      experiment_name=out_experiment)
        rid = make_run_id(out_cfg)
        conf_dir = _confidence_dir(out_cfg)
        os.makedirs(conf_dir, exist_ok=True)

        walls_todo = [w for w in out_cfg.walls
                      if force or not os.path.exists(
                          os.path.join(conf_dir, f"{w}_confidence.png"))]

        models = []
        if walls_todo:
            print(f"=== confidence {out_experiment} {rid}: {n_runs} models "
                  f"from {source_experiment} ({checkpoint_filename}) | "
                  f"device={device} ===")
            for run_n in range(1, n_runs + 1):
                src_cfg = dataclasses.replace(
                    base_config, channels=ch, run_number=run_n,
                    experiment_name=source_experiment,
                    checkpoint_filename=checkpoint_filename)
                ckpt = paths.checkpoint_path(src_cfg)
                if not os.path.exists(ckpt):
                    raise FileNotFoundError(
                        f"ensemble member missing: {ckpt}\n"
                        f"(source_experiment must hold {n_runs} trained runs)")
                model, _, _ = _load_model(ckpt, device)
                models.append(model)

        rows = []
        pooled = {"sum": 0.0, "n": 0, "n_above": 0,
                  "hist": np.zeros(N_HIST_BINS, dtype=np.int64),
                  "sum_stone": 0.0, "n_stone": 0, "n_above_stone": 0,
                  "hist_stone": np.zeros(N_HIST_BINS, dtype=np.int64),
                  "agree": 0, "agree_n": 0}

        for wall in out_cfg.walls:
            conf_path = os.path.join(conf_dir, f"{wall}_confidence.png")

            if os.path.exists(conf_path) and not force:
                print(f"[skip-if-exists] confidence raster present: {conf_path}")
                conf16 = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED)
                if conf16 is None or conf16.dtype != np.uint16:
                    raise ValueError(f"unreadable confidence raster: {conf_path}")
                conf = conf16.astype(np.float32) / 65535.0
                agreement = np.nan
            else:
                class_scores = _wall_probabilities(models, out_cfg, wall, device)
                conf = class_scores.max(axis=2)
                pred = class_scores.argmax(axis=2).astype(np.uint8)

                cv2.imwrite(conf_path,
                            np.round(conf * 65535.0).astype(np.uint16))
                print(f"saved {conf_path}")

                if wall == probs_wall:
                    probs_path = os.path.join(conf_dir, f"{wall}_probs.npz")
                    np.savez_compressed(probs_path,
                                        probs=class_scores.astype(np.float16))
                    print(f"saved {probs_path}")

                saved_rgb = np.array(Image.open(
                    paths.segmentation_raw_path(out_cfg, wall)).convert("RGB"))
                saved_mask = rgb_to_class_mask(saved_rgb)
                if saved_mask.shape != pred.shape:
                    raise ValueError(
                        f"{wall}: shape mismatch recomputed {pred.shape} "
                        f"vs saved raster {saved_mask.shape}")
                agreement = float((pred == saved_mask).mean())
                pooled["agree"] += int((pred == saved_mask).sum())
                pooled["agree_n"] += pred.size
                del class_scores, pred

            gt_rgb = np.array(Image.open(
                paths.test_mask_path(out_cfg, wall)).convert("RGB"))
            gt_mask = rgb_to_class_mask(gt_rgb)
            if gt_mask.shape != conf.shape:
                raise ValueError(f"{wall}: shape mismatch confidence "
                                 f"{conf.shape} vs GT {gt_mask.shape}")
            roi = generate_roi(gt_mask, out_cfg.kernel_radius,
                               out_cfg.roi_operation)
            conf_roi = conf[roi]
            # second basis: GT stone pixels only — matches the drafted
            # "mean prediction confidence for stone pixels" wording
            conf_stone = conf[gt_mask > 0]

            rows.append({
                "run_id": rid, "channels": ch, "wall": wall,
                "n_roi_pixels": int(conf_roi.size),
                "mean_confidence_roi": float(conf_roi.mean()),
                "median_confidence_roi": float(np.median(conf_roi)),
                "frac_above_090_roi": float((conf_roi > CONF_THRESHOLD).mean()),
                "n_stone_pixels": int(conf_stone.size),
                "mean_confidence_stone": float(conf_stone.mean()),
                "median_confidence_stone": float(np.median(conf_stone)),
                "frac_above_090_stone": float((conf_stone > CONF_THRESHOLD).mean()),
                "raster_agreement": agreement,
            })
            pooled["sum"] += float(conf_roi.sum())
            pooled["n"] += int(conf_roi.size)
            pooled["n_above"] += int((conf_roi > CONF_THRESHOLD).sum())
            hist, _ = np.histogram(conf_roi, bins=N_HIST_BINS, range=(0.0, 1.0))
            pooled["hist"] += hist
            pooled["sum_stone"] += float(conf_stone.sum())
            pooled["n_stone"] += int(conf_stone.size)
            pooled["n_above_stone"] += int((conf_stone > CONF_THRESHOLD).sum())
            hist_s, _ = np.histogram(conf_stone, bins=N_HIST_BINS,
                                     range=(0.0, 1.0))
            pooled["hist_stone"] += hist_s
            del conf, conf_roi, conf_stone, gt_mask, roi

        # pooled AllWalls row (pixel-weighted; only meaningful for >1 wall)
        if len(out_cfg.walls) > 1:
            cum = np.cumsum(pooled["hist"])
            median_bin = int(np.searchsorted(cum, pooled["n"] / 2.0))
            cum_s = np.cumsum(pooled["hist_stone"])
            median_bin_s = int(np.searchsorted(cum_s, pooled["n_stone"] / 2.0))
            rows.append({
                "run_id": rid, "channels": ch, "wall": "AllWalls",
                "n_roi_pixels": pooled["n"],
                "mean_confidence_roi": pooled["sum"] / pooled["n"],
                "median_confidence_roi": (median_bin + 0.5) / N_HIST_BINS,
                "frac_above_090_roi": pooled["n_above"] / pooled["n"],
                "n_stone_pixels": pooled["n_stone"],
                "mean_confidence_stone": pooled["sum_stone"] / pooled["n_stone"],
                "median_confidence_stone": (median_bin_s + 0.5) / N_HIST_BINS,
                "frac_above_090_stone": pooled["n_above_stone"] / pooled["n_stone"],
                "raster_agreement": (pooled["agree"] / pooled["agree_n"]
                                     if pooled["agree_n"] else np.nan),
            })

        summary_path = os.path.join(conf_dir, "confidence_summary.csv")
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"saved {summary_path}")

        if models:
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()
