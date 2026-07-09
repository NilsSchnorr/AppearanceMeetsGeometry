"""
Training-history export: pull the per-epoch curves out of saved checkpoints.

Every checkpoint written by train.py embeds its full training history
(train/val loss, accuracy, and batch-IoU per epoch). This module extracts
those dicts from one or more experiments into a single lightweight long-form
CSV, so the curves can be plotted and compared without moving multi-MB
checkpoint files between machines.

Run this WHERE THE CHECKPOINTS LIVE (the training machine). CPU-only and
fast: each checkpoint is torch.load-ed to CPU just to read its "history"
entry. Missing checkpoints (e.g. 3ch runs in a (4,7)-only screen) are skipped
with a note, so one experiment list can cover heterogeneous sweeps.

Output: <experiments_root>/<out_name>/training_histories.csv with columns
    experiment, run_id, channels, run_number, epoch (1-based),
    loss, val_loss, accuracy, val_accuracy, iou_metric, val_iou_metric

Interpretation note: validation metrics are always computed on CLEAN
(unaugmented) tiles, so val curves are directly comparable across all
experiments, including v7_photoaug. TRAIN metrics for photo-augmented runs are
computed on jittered inputs, so their absolute level is expectedly worse than
baseline train metrics — compare their *shape*, not their level.
"""

import dataclasses
import os

import pandas as pd
import torch

from . import paths
from .config import make_run_id

HISTORY_KEYS = ("loss", "val_loss", "accuracy", "val_accuracy",
                "iou_metric", "val_iou_metric")


def export_histories(base_config, experiments, channel_variants=(3, 4, 7),
                     n_runs=5, out_name="v7_histories", force=False):
    """
    Extract training histories from every (experiment, variant, run) checkpoint
    into one combined CSV. Skip-if-exists on the output; missing checkpoints
    are tolerated (printed and skipped).
    """
    out_dir = os.path.join(base_config.experiments_root, out_name)
    out_path = os.path.join(out_dir, "training_histories.csv")
    if os.path.exists(out_path) and not force:
        print(f"[skip-if-exists] histories present: {out_path}")
        return pd.read_csv(out_path)

    rows = []
    for exp in experiments:
        for ch in channel_variants:
            for run_n in range(1, n_runs + 1):
                cfg = dataclasses.replace(base_config, experiment_name=exp,
                                          channels=ch, run_number=run_n)
                rid = make_run_id(cfg)
                ckpt_path = paths.checkpoint_path(cfg)
                if not os.path.exists(ckpt_path):
                    print(f"(missing) {exp} {rid}: {ckpt_path} — skipped")
                    continue
                ckpt = torch.load(ckpt_path, map_location="cpu",
                                  weights_only=False)
                hist = ckpt.get("history")
                if not hist:
                    print(f"(no history) {exp} {rid} — skipped")
                    continue
                n_epochs = len(hist[HISTORY_KEYS[0]])
                for e in range(n_epochs):
                    row = {"experiment": exp, "run_id": rid, "channels": ch,
                           "run_number": run_n, "epoch": e + 1}
                    for k in HISTORY_KEYS:
                        row[k] = float(hist[k][e]) if k in hist else float("nan")
                    rows.append(row)
                print(f"[ok] {exp} {rid}: {n_epochs} epochs")

    if not rows:
        raise FileNotFoundError("no histories found — check experiments/paths")
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(df)} rows, "
          f"{df.groupby(['experiment', 'run_id']).ngroups} runs)")
    return df
