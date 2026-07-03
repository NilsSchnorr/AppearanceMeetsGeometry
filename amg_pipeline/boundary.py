"""
Boundary-quality metric for existing runs: Boundary IoU (Cheng et al. 2021),
computed on the already-saved RAW segmentation rasters. CPU-only, no models,
no GPU — it quantifies the paper's Fig.-7 claim ("the full model produces
cleaner stone boundaries") that pixel IoU barely registers, because boundary
pixels are a small fraction of stone area.

Definition, per stone class c and pixel radius d:
    inner boundary band  B_d(M) = M  AND NOT erode(M, disk(d))
    Boundary IoU(c)      = | B_d(GT_c) ∩ B_d(Pred_c) ∩ ROI |
                           / | (B_d(GT_c) ∪ B_d(Pred_c)) ∩ ROI |
Bands are computed on the full masks first and only then restricted to the
same morphological-closing ROI as the pixel metrics, so out-of-bond
detections are excluded consistently without creating artificial band edges
at the ROI cut. Classes with zero GT pixels inside the ROI are dropped (NaN),
matching evaluate.py's absent-class convention; the AllWalls aggregate
averages each class only over walls where it is present, then averages the
per-class means — identical to the manifest's IoU_mean_stones convention.

The radius d is in PIXELS. Walls differ in mm/px (~1.75-6.25), so absolute
physical band width varies per wall; the three variants are always compared
on the same walls with the same d, so the *comparison* is unaffected. Running
two radii (default 5 and 15 px) shows whether the ranking is d-robust.

GT bands and ROIs depend only on (wall, radius), so they are computed once
and reused across every variant and run — the expensive morphology is not
repeated 15x per wall.
"""

import dataclasses
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from . import paths
from .config import make_run_id
from .data import CLASS_NAMES, rgb_to_class_mask
from .evaluate import STONE_CLASSES, generate_roi

Image.MAX_IMAGE_PIXELS = None


def boundary_band(binary_mask, radius_px):
    """Inner boundary band of a boolean mask: mask AND NOT erode(mask, disk)."""
    if radius_px < 1:
        raise ValueError(f"radius_px must be >= 1, got {radius_px}")
    m8 = binary_mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * radius_px + 1, 2 * radius_px + 1))
    eroded = cv2.erode(m8, k)
    return binary_mask & (eroded == 0)


def boundary_iou_per_class(gt_mask, pred_mask, roi, gt_bands, radius_px):
    """
    Per-class Boundary IoU for one (wall, prediction, radius).
    gt_bands: precomputed {class_name: band} for this wall+radius.
    Returns {class_name: float or NaN} over the three stone classes.
    """
    scores = {}
    for idx, name in enumerate(CLASS_NAMES):
        if name == "Background":
            continue
        if name not in gt_bands:          # class absent on this wall -> NaN
            scores[name] = np.nan
            continue
        pred_band = boundary_band(pred_mask == idx, radius_px) & roi
        gt_band = gt_bands[name]          # already ROI-restricted
        inter = np.count_nonzero(gt_band & pred_band)
        union = np.count_nonzero(gt_band | pred_band)
        scores[name] = (inter / union) if union > 0 else np.nan
    return scores


def _mean_stones(scores):
    vals = [scores[c] for c in STONE_CLASSES if not np.isnan(scores[c])]
    return float(np.mean(vals)) if vals else np.nan


def run_boundary_eval(base_config, frozen_experiment, out_name="v6_boundary",
                      channel_variants=(3, 4, 7), n_runs=5, radii=(5, 15),
                      force=False):
    """
    Compute Boundary IoU for every (variant, run, wall, radius) of an existing
    experiment's saved RAW rasters, and write one manifest:
        <experiments_root>/<out_name>/boundary_manifest.csv
    Rows mirror the master-manifest schema (plus 'radius_px' and BIoU_ columns)
    and include an AllWalls aggregate per (run, radius). The source experiment
    is only read, never written.
    """
    out_dir = os.path.join(base_config.experiments_root, out_name)
    out_path = os.path.join(out_dir, "boundary_manifest.csv")
    if os.path.exists(out_path) and not force:
        print(f"[skip-if-exists] boundary manifest present: {out_path}")
        return pd.read_csv(out_path)

    src = dataclasses.replace(base_config, experiment_name=frozen_experiment)
    rows = []

    for wall in src.walls:
        print(f"=== {wall}: GT mask, ROI, GT bands (shared across all runs) ===")
        gt_rgb = np.array(Image.open(paths.test_mask_path(src, wall)).convert("RGB"))
        gt_mask = rgb_to_class_mask(gt_rgb)
        roi = generate_roi(gt_mask, src.kernel_radius, src.roi_operation)

        # Precompute GT bands per radius, only for classes present in the ROI.
        gt_bands = {}
        for r in radii:
            bands = {}
            for idx, name in enumerate(CLASS_NAMES):
                if name == "Background":
                    continue
                gt_bin = (gt_mask == idx)
                if not np.count_nonzero(gt_bin & roi):
                    continue  # absent on this wall -> stays NaN downstream
                bands[name] = boundary_band(gt_bin, r) & roi
            gt_bands[r] = bands

        for ch in channel_variants:
            for run_n in range(1, n_runs + 1):
                cfg = dataclasses.replace(src, channels=ch, run_number=run_n)
                rid = make_run_id(cfg)
                pred_path = paths.segmentation_raw_path(cfg, wall)
                if not os.path.exists(pred_path):
                    print(f"(missing) {rid} {wall}: {pred_path} — skipped")
                    continue
                pred_rgb = np.array(Image.open(pred_path).convert("RGB"))
                pred_mask = rgb_to_class_mask(pred_rgb)
                if pred_mask.shape != gt_mask.shape:
                    raise ValueError(f"{rid} {wall}: shape mismatch "
                                     f"pred {pred_mask.shape} vs GT {gt_mask.shape}")
                per_radius = {}
                for r in radii:
                    scores = boundary_iou_per_class(gt_mask, pred_mask, roi,
                                                    gt_bands[r], r)
                    per_radius[r] = scores
                    rows.append({
                        "run_id": rid, "channels": ch, "run_number": run_n,
                        "experiment": frozen_experiment, "wall": wall,
                        "radius_px": r,
                        "BIoU_Ashlar": scores["Ashlar"],
                        "BIoU_Polygonal": scores["Polygonal"],
                        "BIoU_Quarry": scores["Quarry"],
                        "BIoU_mean_stones": _mean_stones(scores),
                    })
                print(f"  {rid} {wall}: " + " | ".join(
                    f"d={r}px BIoU={_mean_stones(per_radius[r]):.3f}" for r in radii))

    df = pd.DataFrame(rows)

    # ---- AllWalls aggregate (per class: mean over walls where present) ------
    agg_rows = []
    for (ch, run_n, r), g in df.groupby(["channels", "run_number", "radius_px"]):
        agg = {}
        for c in STONE_CLASSES:
            vals = g[f"BIoU_{c}"].dropna()
            agg[c] = float(vals.mean()) if len(vals) else np.nan
        agg_rows.append({
            "run_id": g["run_id"].iloc[0], "channels": ch, "run_number": run_n,
            "experiment": frozen_experiment, "wall": "AllWalls", "radius_px": r,
            "BIoU_Ashlar": agg["Ashlar"], "BIoU_Polygonal": agg["Polygonal"],
            "BIoU_Quarry": agg["Quarry"],
            "BIoU_mean_stones": float(np.nanmean([agg[c] for c in STONE_CLASSES])),
        })
    df = pd.concat([df, pd.DataFrame(agg_rows)], ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(df)} rows)")
    return df
