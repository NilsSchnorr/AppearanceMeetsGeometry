"""
Ensemble inference: average the softmax probabilities of all N runs of a
variant and evaluate the result as one "model" through the unchanged pipeline.

Rationale: five trained checkpoints per variant already exist for every
experiment; averaging their per-window probabilities is a standard, honest
technique that typically improves both accuracy and boundary stability at ZERO
training cost. Applied identically to all three variants, it keeps the
3ch/4ch/7ch comparison fair while lifting every absolute number.

Mechanics — mirrors segment.py exactly (same windowing, same per-window
preprocessing, same center-weighted merge): the ONLY difference is that each
window's class probabilities are the mean over the N models' softmax outputs.
Outputs land in their own experiment (`out_experiment`) with run_id
"<ch>ch_run1" so the standard evaluator and manifest machinery apply
unchanged; provenance columns (source_experiment, n_models,
checkpoint_filename) are added to the manifest rows.

checkpoint_filename="model_best.pth" ensembles the best-val checkpoints
instead of the final-epoch ones (both are saved by train.py since C3).

Note on test-time augmentation: deliberately NOT implemented here. A mirrored
input requires negating the horizontal component of the normal map (the
training pipeline avoided exactly this by flipping BEFORE normal-map
generation); doing it at inference needs the generator's channel/sign
convention verified first. Ensemble-only keeps this module convention-free.
"""

import dataclasses
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from . import paths
from .config import make_run_id
from .data import CLASS_COLORS_RGB
from .evaluate import append_to_manifest, evaluate
from .segment import (_build_wall_stack, _load_model, _segment_window,
                      combine_center_weighted, create_sliding_windows,
                      simple_cleanup)


def run_ensemble(base_config, source_experiment, out_experiment,
                 channel_variants=(3, 4, 7), n_runs=5,
                 checkpoint_filename="model.pth", do_evaluate=True,
                 force=False):
    """
    For each variant: load the N run checkpoints of `source_experiment`,
    segment every test wall with mean-softmax fusion, save rasters under
    `out_experiment` (run_id "<ch>ch_run1"), and evaluate + append to that
    experiment's manifest. Skip-if-exists per wall on the RAW raster; the
    source experiment is only ever read.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ch in channel_variants:
        out_cfg = dataclasses.replace(base_config, channels=ch, run_number=1,
                                      experiment_name=out_experiment)
        rid = make_run_id(out_cfg)
        walls_todo = [w for w in out_cfg.walls
                      if force or not os.path.exists(paths.segmentation_raw_path(out_cfg, w))]

        models, n_classes = [], None
        if walls_todo:
            print(f"=== ensemble {out_experiment} {rid}: {n_runs} models from "
                  f"{source_experiment} ({checkpoint_filename}) | device={device} ===")
            for run_n in range(1, n_runs + 1):
                src_cfg = dataclasses.replace(base_config, channels=ch, run_number=run_n,
                                              experiment_name=source_experiment,
                                              checkpoint_filename=checkpoint_filename)
                ckpt = paths.checkpoint_path(src_cfg)
                if not os.path.exists(ckpt):
                    raise FileNotFoundError(
                        f"ensemble member missing: {ckpt}\n"
                        f"(source_experiment must hold {n_runs} trained runs)")
                model, n_classes, img_ch = _load_model(ckpt, device)
                models.append(model)
            os.makedirs(paths.segmentation_dir(out_cfg), exist_ok=True)

        colors = np.array(CLASS_COLORS_RGB)
        for wall in out_cfg.walls:
            raw_path = paths.segmentation_raw_path(out_cfg, wall)
            if os.path.exists(raw_path) and not force:
                print(f"[skip-if-exists] ensemble raster present: {raw_path}")
                continue
            stack = _build_wall_stack(out_cfg, wall)
            windows, positions = create_sliding_windows(stack, out_cfg.window_size,
                                                        out_cfg.stride)
            windows_data = []
            for i, win in enumerate(tqdm(windows, desc=f"{wall}: windows x{len(models)} models")):
                probs_sum = None
                for m in models:
                    _, probs = _segment_window(m, win, out_cfg.model_size,
                                               out_cfg.window_size, device)
                    probs_sum = probs if probs_sum is None else probs_sum + probs
                probs_mean = probs_sum / float(len(models))
                pred = np.argmax(probs_mean, axis=0).astype(np.uint8)
                windows_data.append((pred, probs_mean))
                if i % 10 == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()

            seg = combine_center_weighted(windows_data, positions, stack.shape,
                                          n_classes, out_cfg.window_size)
            colored = np.zeros((*seg.shape, 3), dtype=np.uint8)
            for i in range(n_classes):
                colored[seg == i] = colors[i]
            cv2.imwrite(raw_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
            cv2.imwrite(paths.segmentation_clean_path(out_cfg, wall), simple_cleanup(seg))
            print(f"saved {raw_path}")

        if models:
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if do_evaluate:
            _, rows = evaluate(out_cfg, force=force)
            for r in rows:  # provenance for the decision table
                r["source_experiment"] = source_experiment
                r["n_models"] = n_runs
                r["checkpoint_filename"] = checkpoint_filename
            append_to_manifest(out_cfg, rows)
