"""
evaluate(config) — computes ROI-based metrics for one run (one variant) across
all test walls, faithfully reproducing 04_segmentation_evaluation/
ROI_segmentation_evaluation.ipynb.

Per wall it:
  - loads the GT mask and builds the ROI by morphological closing/dilation of
    the stone mask (kernel_radius), so out-of-bond detections are excluded;
  - loads this run's RAW prediction raster;
  - computes per-class IoU (NaN for classes absent from the wall) and
    precision/recall/F1/support within the ROI;
  - writes the brief's per-wall, per-variant detail CSV
    (Class, IoU, Precision, Recall, F1-Score, Support) and a per-run summary CSV.

It returns manifest rows (one per wall + one "AllWalls" aggregate) and the
orchestrator appends them idempotently to the single master manifest.csv.

Mean-IoU over stone classes excludes Background and absent (NaN) classes, and
the AllWalls aggregate averages each class only over walls where it is present
— matching the agreed MC-ROI averaging (no zero-padding of absent classes).
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.morphology import disk, binary_closing, binary_dilation
from sklearn.metrics import classification_report

from .data import rgb_to_class_mask, CLASS_NAMES
from .config import make_run_id
from . import paths

Image.MAX_IMAGE_PIXELS = None
STONE_CLASSES = ["Ashlar", "Polygonal", "Quarry"]
VARIANT_TAG = {3: "3_channel", 4: "4_channel", 7: "7_channel"}


def generate_roi(gt_class_mask, kernel_radius, operation="closing"):
    stone = (gt_class_mask > 0).astype(np.uint8)
    selem = disk(kernel_radius)
    roi = binary_dilation(stone, selem) if operation == "dilation" else binary_closing(stone, selem)
    return roi.astype(bool)


def metrics_within_roi(gt_mask, pred_mask, roi_mask):
    roi_flat = roi_mask.flatten()
    gt_flat = gt_mask.flatten()[roi_flat]
    pred_flat = pred_mask.flatten()[roi_flat]

    iou_scores = {}
    for idx, name in enumerate(CLASS_NAMES):
        gt_b = (gt_flat == idx)
        # A class is ABSENT from this wall iff it has zero GT pixels in the ROI.
        # Drop it (NaN) rather than scoring it 0.0: a model that hallucinates a few
        # pixels of an absent class would otherwise make union>0 and produce a
        # spurious 0.0 that deflates the mean. Those false positives are still
        # penalised, because they reduce the IoU of the present class they overlap.
        if gt_b.sum() == 0:
            iou_scores[name] = np.nan
            continue
        pred_b = (pred_flat == idx)
        inter = np.logical_and(gt_b, pred_b).sum()
        union = np.logical_or(gt_b, pred_b).sum()  # > 0 guaranteed (gt_b.sum() > 0)
        iou_scores[name] = inter / union

    report = classification_report(
        gt_flat, pred_flat, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    f1 = {}
    for name in CLASS_NAMES:
        r = report.get(name, {})
        support = r.get("support", 0)
        absent = (support == 0)  # same absence rule as IoU: zero GT support
        f1[name] = {"precision": np.nan if absent else r.get("precision", np.nan),
                    "recall": np.nan if absent else r.get("recall", np.nan),
                    "f1-score": np.nan if absent else r.get("f1-score", np.nan),
                    "support": support}
    # macro_F1 = mean F1 over the stone classes that are present (NaN-aware),
    # excluding Background and absent classes — consistent with IoU_mean_stones.
    stone_f1 = [f1[c]["f1-score"] for c in STONE_CLASSES]
    f1["macro_avg"] = float(np.nanmean(stone_f1)) if not all(np.isnan(stone_f1)) else np.nan
    f1["weighted_avg"] = report["weighted avg"]["f1-score"]
    return iou_scores, f1


def _mean_stone_iou(iou_scores):
    vals = [iou_scores[c] for c in STONE_CLASSES if not np.isnan(iou_scores[c])]
    return float(np.mean(vals)) if vals else np.nan


def evaluate(config, force=False):
    """Evaluate all walls for this run. Returns (per_wall_results, manifest_rows)."""
    run_id = make_run_id(config)
    per_wall = {}
    manifest_rows = []
    op = config.roi_operation
    variant = VARIANT_TAG[config.channels]

    for wall in config.walls:
        wall_dir = paths.metrics_wall_dir(config, wall)
        detail_path = os.path.join(wall_dir, f"roi_evaluation_{variant}_{op}.csv")
        if os.path.exists(detail_path) and not force:
            print(f"[skip-if-exists] metrics present: {detail_path}")
            detail_df = pd.read_csv(detail_path)
            iou_scores = {r["Class"]: r["IoU"] for _, r in detail_df.iterrows()}
            f1 = {r["Class"]: {"f1-score": r["F1-Score"]} for _, r in detail_df.iterrows()}
            macro = float(np.nanmean([f1[c]["f1-score"] for c in STONE_CLASSES]))
        else:
            gt_rgb = np.array(Image.open(paths.test_mask_path(config, wall)).convert("RGB"))
            gt_mask = rgb_to_class_mask(gt_rgb)
            roi = generate_roi(gt_mask, config.kernel_radius, op)

            pred_rgb = np.array(Image.open(paths.segmentation_raw_path(config, wall)).convert("RGB"))
            pred_mask = rgb_to_class_mask(pred_rgb)
            if pred_mask.shape != gt_mask.shape:
                raise ValueError(f"{wall}: shape mismatch pred {pred_mask.shape} vs GT {gt_mask.shape}")

            iou_scores, f1 = metrics_within_roi(gt_mask, pred_mask, roi)
            macro = f1["macro_avg"]

            os.makedirs(wall_dir, exist_ok=True)
            rows = [{
                "Class": c,
                "IoU": iou_scores[c],
                "Precision": f1[c]["precision"],
                "Recall": f1[c]["recall"],
                "F1-Score": f1[c]["f1-score"],
                "Support": f1[c]["support"],
            } for c in CLASS_NAMES]
            detail_df = pd.DataFrame(rows)
            detail_df.to_csv(detail_path, index=False)
            print(f"saved {detail_path}")

        per_wall[wall] = {"iou": iou_scores, "mean_stone_iou": _mean_stone_iou(iou_scores),
                          "macro_f1": macro}

        manifest_rows.append({
            "run_id": run_id, "channels": config.channels, "run_number": config.run_number,
            "width_mult": config.width_mult, "seed": config.seed,
            "experiment": config.experiment_name, "wall": wall,
            "IoU_Ashlar": iou_scores["Ashlar"], "IoU_Polygonal": iou_scores["Polygonal"],
            "IoU_Quarry": iou_scores["Quarry"], "IoU_mean_stones": _mean_stone_iou(iou_scores),
            "F1_Ashlar": f1["Ashlar"].get("f1-score", np.nan),
            "F1_Polygonal": f1["Polygonal"].get("f1-score", np.nan),
            "F1_Quarry": f1["Quarry"].get("f1-score", np.nan),
            "macro_F1": macro,
        })

    # ---- AllWalls aggregate row (per-class mean over walls where present) ---
    def _agg(key):
        vals = [per_wall[w]["iou"][key] for w in config.walls
                if not np.isnan(per_wall[w]["iou"][key])]
        return float(np.mean(vals)) if vals else np.nan

    agg_iou = {c: _agg(c) for c in CLASS_NAMES}
    manifest_rows.append({
        "run_id": run_id, "channels": config.channels, "run_number": config.run_number,
        "width_mult": config.width_mult, "seed": config.seed,
        "experiment": config.experiment_name, "wall": "AllWalls",
        "IoU_Ashlar": agg_iou["Ashlar"], "IoU_Polygonal": agg_iou["Polygonal"],
        "IoU_Quarry": agg_iou["Quarry"],
        "IoU_mean_stones": float(np.nanmean([agg_iou[c] for c in STONE_CLASSES])),
        "F1_Ashlar": np.nan, "F1_Polygonal": np.nan, "F1_Quarry": np.nan,
        "macro_F1": float(np.nanmean([per_wall[w]["macro_f1"] for w in config.walls])),
    })

    # ---- per-run summary CSV (one row per wall + AllWalls) ------------------
    summary_path = os.path.join(paths.metrics_dir(config), f"roi_summary_{variant}_{op}.csv")
    os.makedirs(paths.metrics_dir(config), exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(summary_path, index=False)
    print(f"saved {summary_path}")

    return per_wall, manifest_rows


def append_to_manifest(config, manifest_rows):
    """Idempotently merge rows into the single master manifest.csv (resume-safe)."""
    mpath = paths.manifest_path(config)
    new_df = pd.DataFrame(manifest_rows)
    if os.path.exists(mpath):
        old = pd.read_csv(mpath)
        # drop any existing rows for these (run_id, wall) pairs, then append fresh
        keys = set(zip(new_df["run_id"], new_df["wall"]))
        mask = [(r, w) not in keys for r, w in zip(old["run_id"], old["wall"])]
        merged = pd.concat([old[mask], new_df], ignore_index=True)
    else:
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        merged = new_df
    merged.to_csv(mpath, index=False)
    print(f"manifest -> {mpath} ({len(merged)} rows)")
    return mpath
