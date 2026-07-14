"""
gapmetrics.py — Step 5: inter-stone gap detection + stone-level separation.

Faithful port of the two archived evaluation notebooks
(Archive/04_segmentation_evaluation/ROI_gap_segmentation_evaluation.ipynb and
stone_detection_evaluation.ipynb) into pipeline conventions. These are the
metrics behind the paper's stone-boundary claims: pixel IoU cannot distinguish
a model that separates stones from one that merges them, because the gap
pixels are a tiny fraction of the wall.

GAP METRIC (per wall):
  gaps := binary_closing(stone_mask, disk(kernel_radius)) & ~stone_mask.
  GT gaps come from the GT stone mask; the GT closed mask is the ROI.
  Predicted gaps are computed the same way from the prediction, restricted to
  the ROI. Reported: gap IoU / precision / recall / F1 + pixel counts and the
  predicted/GT gap-pixel ratio (>1.2 over-segments, <0.8 under-segments).

STONE METRIC (bidirectional, per GT stone):
  coverage    = fraction of the GT stone covered by SAME-class predictions
                (fragments summed — fragmentation alone is not penalised);
  separation  = overlap-weighted mean "commitment" of the overlapping
                same-class predicted components, where commitment =
                (overlap with this stone) / (component's total size);
                a component spanning two stones has low commitment -> merge;
  status      : detected  (coverage >= t_cov AND separation >= t_sep)
                merged    (coverage >= t_cov AND separation <  t_sep)
                undetected(coverage <  t_cov).
  Statuses are derived from (coverage, separation) after the fact, so one
  pass emits counts for several threshold pairs (columns suffixed _cXX_sYY).

Implementation notes (numerics identical to the notebooks):
  - rgb_to_class_mask is the shared pipeline implementation (data.py);
  - GT stones store bbox-local masks instead of full-frame masks (memory);
  - prediction components are extracted from the raw prediction (not
    ROI-masked), exactly as in the notebook — out-of-bond extensions of a
    component correctly lower its commitment;
  - the "AllWalls" row POOLS stones and gap-pixel counts across walls
    (stone counts are the natural weights), rather than averaging wall means.

Not wired into config.eval_type (which stays reserved); orchestrated directly
via run_gap_eval() from a notebook. Eval-only: reads GT masks + RAW rasters.
"""

import dataclasses
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.measure import regionprops
from skimage.morphology import disk, binary_closing

from . import paths
from .config import make_run_id
from .data import rgb_to_class_mask, CLASS_NAMES

Image.MAX_IMAGE_PIXELS = None
STONE_CLASS_NAMES = {1: "Ashlar", 2: "Polygonal", 3: "Quarry"}
DEFAULT_THRESHOLD_PAIRS = ((0.9, 0.9), (0.7, 0.7), (0.5, 0.5))


# --------------------------------------------------------------------------
# gap metric
# --------------------------------------------------------------------------
def extract_gaps_via_closing(binary_mask, kernel_radius, roi_mask=None):
    """Verbatim notebook logic: gaps = closing(mask) & ~mask (within ROI)."""
    working = binary_mask * roi_mask if roi_mask is not None else binary_mask
    closed = binary_closing(working.astype(bool), disk(kernel_radius))
    gaps = closed & ~working.astype(bool)
    if roi_mask is not None:
        gaps = gaps & roi_mask
    return closed, gaps


def gap_pixel_counts(gt_gaps, pred_gaps):
    gt = gt_gaps.astype(bool)
    pr = pred_gaps.astype(bool)
    tp = int(np.sum(gt & pr))
    fp = int(np.sum(~gt & pr))
    fn = int(np.sum(gt & ~pr))
    return tp, fp, fn


def gap_metrics_from_counts(tp, fp, fn):
    union = tp + fp + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"gap_iou": tp / union if union else 0.0,
            "gap_precision": prec, "gap_recall": rec, "gap_f1": f1,
            "gap_tp": tp, "gap_fp": fp, "gap_fn": fn,
            "gt_gap_pixels": tp + fn, "pred_gap_pixels": tp + fp,
            "gap_ratio": (tp + fp) / (tp + fn) if (tp + fn) else 0.0}


# --------------------------------------------------------------------------
# stone metric
# --------------------------------------------------------------------------
def extract_stones(class_mask, min_size=100):
    """GT stones as connected components of the stone binary; class by pixel
    majority. Bbox-local masks (memory-safe, numerics identical)."""
    labeled, _ = ndimage.label((class_mask > 0).astype(np.uint8))
    stones = []
    for region in regionprops(labeled):
        if region.area < min_size:
            continue
        local = labeled[region.slice] == region.label
        classes = class_mask[region.slice][local]
        class_id = int(np.bincount(classes).argmax())
        stones.append({"stone_id": int(region.label), "class_id": class_id,
                       "pixel_count": int(region.area),
                       "slice": region.slice, "local_mask": local})
    return stones


def extract_prediction_regions(pred_mask, min_size=10):
    """Class-agnostic connected components of the prediction; small regions
    zeroed; component class by majority over its non-background pixels."""
    labeled, _ = ndimage.label((pred_mask > 0).astype(np.uint8))
    pred_info = {}
    for region in regionprops(labeled):
        if region.area < min_size:
            labeled[labeled == region.label] = 0
            continue
        local = labeled[region.slice] == region.label
        classes = pred_mask[region.slice][local]
        nz = classes[classes > 0]
        class_id = int(np.bincount(nz).argmax()) if nz.size else 0
        pred_info[int(region.label)] = {"class_id": class_id,
                                        "pixel_count": int(region.area)}
    return labeled, pred_info


def evaluate_stone(stone, pred_mask, labeled_preds, pred_info):
    """Coverage + separation for one GT stone (status assigned later)."""
    sl, local = stone["slice"], stone["local_mask"]
    gt_class, gt_pixels = stone["class_id"], stone["pixel_count"]
    pred_within = pred_mask[sl][local]
    labels_within = labeled_preds[sl][local]

    same = int(np.sum(pred_within == gt_class))
    wrong = int(np.sum((pred_within > 0) & (pred_within != gt_class)))
    coverage = same / gt_pixels
    wrong_class_coverage = wrong / gt_pixels
    uncovered = int(np.sum(pred_within == 0)) / gt_pixels

    labels = np.unique(labels_within)
    labels = labels[labels > 0]
    same_class_labels = [l for l in labels
                         if l in pred_info and pred_info[l]["class_id"] == gt_class]
    if not same_class_labels:
        separation, num_preds = 0.0, 0
    else:
        total_w = weighted_sum = 0.0
        for lbl in same_class_labels:
            overlap = float(np.sum(labels_within == lbl))
            commitment = overlap / pred_info[lbl]["pixel_count"]
            total_w += overlap
            weighted_sum += overlap * commitment
        separation = weighted_sum / total_w if total_w > 0 else 0.0
        num_preds = len(same_class_labels)

    return {"stone_id": stone["stone_id"], "class_id": gt_class,
            "class_name": STONE_CLASS_NAMES.get(gt_class, str(gt_class)),
            "pixel_count": gt_pixels, "coverage": coverage,
            "wrong_class_coverage": wrong_class_coverage,
            "uncovered": uncovered, "separation": separation,
            "num_predictions": num_preds}


def _status(e, t_cov, t_sep):
    if e["coverage"] >= t_cov:
        return "detected" if e["separation"] >= t_sep else "merged"
    return "undetected"


def stone_summary(evals, threshold_pairs=DEFAULT_THRESHOLD_PAIRS):
    """Aggregate a list of per-stone evaluations into one row of columns."""
    n = len(evals)
    out = {"n_stones": n}
    if n == 0:
        return out
    cov = np.array([e["coverage"] for e in evals])
    sep = np.array([e["separation"] for e in evals])
    out["mean_coverage"] = float(cov.mean())
    covered = cov > 0
    out["mean_separation"] = float(sep[covered].mean()) if covered.any() else 0.0
    out["mean_wrong_class_coverage"] = float(
        np.mean([e["wrong_class_coverage"] for e in evals]))

    for t_cov, t_sep in threshold_pairs:
        tag = f"c{int(round(t_cov*100))}_s{int(round(t_sep*100))}"
        st = [_status(e, t_cov, t_sep) for e in evals]
        det, mrg = st.count("detected"), st.count("merged")
        out[f"detected_{tag}"] = det
        out[f"merged_{tag}"] = mrg
        out[f"undetected_{tag}"] = n - det - mrg
        out[f"detection_rate_{tag}"] = det / n
        out[f"merge_rate_{tag}"] = mrg / n

    # per-class breakdown at the primary threshold pair
    t_cov, t_sep = threshold_pairs[0]
    tag = f"c{int(round(t_cov*100))}_s{int(round(t_sep*100))}"
    for cid, cname in STONE_CLASS_NAMES.items():
        ce = [e for e in evals if e["class_id"] == cid]
        if not ce:
            continue
        cst = [_status(e, t_cov, t_sep) for e in ce]
        out[f"{cname}_n"] = len(ce)
        out[f"{cname}_detection_rate_{tag}"] = cst.count("detected") / len(ce)
        out[f"{cname}_merge_rate_{tag}"] = cst.count("merged") / len(ce)
        out[f"{cname}_mean_coverage"] = float(np.mean([e["coverage"] for e in ce]))
        ccov = [e for e in ce if e["coverage"] > 0]
        out[f"{cname}_mean_separation"] = (float(np.mean([e["separation"] for e in ccov]))
                                           if ccov else 0.0)
    return out


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def _load_class_mask(path):
    return rgb_to_class_mask(np.array(Image.open(path).convert("RGB")))


def _crop_common(a, b, label=""):
    """Defensive: crop both to common min dims (prints if they differ)."""
    if a.shape == b.shape:
        return a, b
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    print(f"    (note) {label}: shape {a.shape} vs {b.shape} -> cropping to {(h, w)}")
    return a[:h, :w], b[:h, :w]


def run_gap_eval(base_config, experiments, channel_variants=(3, 4, 7),
                 out_name="v9_gapstone", kernel_radius=None, min_stone_size=100,
                 threshold_pairs=DEFAULT_THRESHOLD_PAIRS, save_details=False,
                 force=False):
    """
    Compute gap + stone metrics for every (experiment, variant, run, wall).

    experiments: iterable of (experiment_name, n_runs) tuples, e.g.
        [("v2_baseline_ens", 1), ("v2_yaw_correction-epsV1", 5)].
    Missing rasters are skipped with a note. Output: one combined CSV at
    <experiments_root>/<out_name>/gap_stone_metrics.csv with per-wall rows and
    a pooled "AllWalls" row per run (stones pooled, gap counts summed).
    Crash-safe: progress is checkpointed to <name>.partial.csv after each
    experiment and renamed to the final CSV on completion, so skip-if-exists
    (on the final CSV only) never latches onto a partial result. Optional
    per-stone detail CSVs (save_details).
    """
    kr = base_config.kernel_radius if kernel_radius is None else kernel_radius
    out_dir = os.path.join(base_config.experiments_root, out_name)
    out_csv = os.path.join(out_dir, "gap_stone_metrics.csv")
    if os.path.exists(out_csv) and not force:
        print(f"[skip-if-exists] gap/stone metrics present: {out_csv}")
        return pd.read_csv(out_csv)
    os.makedirs(out_dir, exist_ok=True)

    # ---- GT cache: one closing + one stone extraction per wall ----
    gt_cache = {}
    for wall in base_config.walls:
        gt_mask = _load_class_mask(paths.test_mask_path(base_config, wall))
        gt_binary = (gt_mask > 0).astype(np.uint8)
        gt_closed, gt_gaps = extract_gaps_via_closing(gt_binary, kr)
        gt_cache[wall] = {"mask": gt_mask, "roi": gt_closed.astype(bool),
                          "gaps": gt_gaps,
                          "stones": extract_stones(gt_mask, min_stone_size)}
        print(f"[GT] {wall}: {len(gt_cache[wall]['stones'])} stones "
              f"(>= {min_stone_size} px), {int(gt_gaps.sum()):,} gap pixels "
              f"(kernel r={kr})")

    rows = []
    for exp, n_runs in experiments:
        for ch in channel_variants:
            for run_n in range(1, n_runs + 1):
                cfg = dataclasses.replace(base_config, experiment_name=exp,
                                          channels=ch, run_number=run_n)
                rid = make_run_id(cfg)
                run_evals, run_counts = [], np.zeros(3, dtype=np.int64)
                any_wall = False
                for wall in cfg.walls:
                    raw = paths.segmentation_raw_path(cfg, wall)
                    if not os.path.exists(raw):
                        print(f"(missing) {exp} {rid} {wall} — skipped")
                        continue
                    any_wall = True
                    g = gt_cache[wall]
                    pred_mask = _load_class_mask(raw)
                    pred_mask, gt_mask = _crop_common(pred_mask, g["mask"],
                                                      f"{exp} {rid} {wall}")
                    if pred_mask.shape != g["mask"].shape:
                        # re-derive GT products on the cropped frame
                        gt_binary = (gt_mask > 0).astype(np.uint8)
                        roi, gt_gaps = extract_gaps_via_closing(gt_binary, kr)
                        roi = roi.astype(bool)
                        stones = extract_stones(gt_mask, min_stone_size)
                    else:
                        roi, gt_gaps, stones = g["roi"], g["gaps"], g["stones"]

                    pred_binary = (pred_mask > 0).astype(np.uint8)
                    _, pred_gaps = extract_gaps_via_closing(pred_binary, kr, roi)
                    tp, fp, fn = gap_pixel_counts(gt_gaps, pred_gaps)
                    run_counts += (tp, fp, fn)

                    labeled_preds, pred_info = extract_prediction_regions(pred_mask)
                    evals = [evaluate_stone(s, pred_mask, labeled_preds, pred_info)
                             for s in stones]
                    run_evals.extend(evals)

                    row = {"experiment": exp, "run_id": rid, "channels": ch,
                           "run_number": run_n, "wall": wall, "kernel_radius": kr,
                           **gap_metrics_from_counts(tp, fp, fn),
                           **stone_summary(evals, threshold_pairs)}
                    rows.append(row)
                    print(f"[ok] {exp} {rid} {wall}: gap IoU "
                          f"{row['gap_iou']:.3f} | det@{int(threshold_pairs[0][0]*100)}/"
                          f"{int(threshold_pairs[0][1]*100)} "
                          f"{row.get('detection_rate_c90_s90', float('nan')):.3f}")

                    if save_details:
                        det_dir = os.path.join(out_dir, "details", exp, rid)
                        os.makedirs(det_dir, exist_ok=True)
                        pd.DataFrame(evals).drop(columns=[], errors="ignore").to_csv(
                            os.path.join(det_dir, f"{wall}_stones.csv"), index=False)

                if any_wall:  # pooled AllWalls row for this run
                    rows.append({"experiment": exp, "run_id": rid, "channels": ch,
                                 "run_number": run_n, "wall": "AllWalls",
                                 "kernel_radius": kr,
                                 **gap_metrics_from_counts(*run_counts),
                                 **stone_summary(run_evals, threshold_pairs)})
        pd.DataFrame(rows).to_csv(out_csv + ".partial.csv", index=False)
        print(f"[checkpoint] {exp} done ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    if os.path.exists(out_csv + ".partial.csv"):
        os.remove(out_csv + ".partial.csv")
    print(f"saved {out_csv} ({len(df)} rows)")
    return df
