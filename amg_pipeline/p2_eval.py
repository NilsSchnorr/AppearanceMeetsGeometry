"""
p2_eval.py — per-run evaluation for the P2 gap/joint queue (Nov 2026).

For one run (experiment, run_id) and every test wall it writes, next to the
existing outputs:
  1. ROI metrics          via evaluate() + append_to_manifest() (unchanged)
  2. gap + stone metrics  with the paper's gapmetrics functions (gap IoU,
                          precision, recall, gap ratio; coverage/separation),
                          plus stone_area_ratio = predicted stone pixels in the
                          ROI / GT stone pixels
  3. separation measure   (the P2 "Maßstab"): for every GT stone, the
                          class-agnostic connected components of the predicted
                          stone mask (4- and 8-connectivity) that cover at least
                          FRAC (10 %) of it:
                            undetected : none
                            split      : two or more   ("zerfallen")
                            merged     : one of them also covers >= FRAC of
                                         another GT stone ("verschmolzen")
                            alone      : exactly one, and not merged
                          split and merged can occur together; alone excludes
                          both. Two definitions of a GT stone:
                            "mask"    : connected components of the GT mask
                                        >= 100 px (as the paper's stone metric)
                            "polygon" : COCO polygons >= 100 px (stones that
                                        touch in the GT stay separate)
GT products are cached per process. Nothing here changes evaluate.py or
gapmetrics.py; the queue may patch their binary_closing with the exact fast
version (p2_morph) after a pixel-identity self-check.
"""

import os
import json

import cv2
import numpy as np
import pandas as pd

from . import paths
from . import gapmetrics as gm
from .config import make_run_id
from .data import CLASS_NAMES
from .evaluate import evaluate, append_to_manifest

MIN_STONE = 100
FRAC = 0.10
CONNS = (4, 8)
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}   # Background 0, Ashlar 1, Polygonal 2, Quarry 3
_GT_CACHE = {}
_COCO_CACHE = {}


# ---------------------------------------------------------------------------
# GT
# ---------------------------------------------------------------------------
def _rle_counts(s):
    """Decompress a COCO RLE string (rleFrString in maskApi.c)."""
    if isinstance(s, str):
        s = s.encode()
    cnts, p = [], 0
    while p < len(s):
        x = k = 0
        more = 1
        while more:
            c = s[p] - 48
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def polygon_instance_raster(coco_json, file_name, h, w):
    """Instance raster (int32) of the COCO polygons of one image, painted in
    JSON order (the later polygon wins), rasterised exactly like annToMask."""
    from pycocotools import mask as mu
    if coco_json not in _COCO_CACHE:
        with open(coco_json) as f:
            _COCO_CACHE[coco_json] = json.load(f)
    d = _COCO_CACHE[coco_json]
    imgs = [i for i in d["images"] if i["file_name"] == file_name]
    if not imgs:
        raise KeyError(f"{file_name} not in {coco_json}")
    img = imgs[0]
    if (img["height"], img["width"]) != (h, w):
        raise ValueError(f"{file_name}: COCO size {(img['height'], img['width'])} != mask {(h, w)}")
    cat = {c["id"]: c["name"] for c in d["categories"]}
    anns = [a for a in d["annotations"] if a["image_id"] == img["id"]]
    flat = np.zeros(h * w, dtype=np.int32)
    cls = [0]
    for i, a in enumerate(anns, start=1):
        seg = a["segmentation"]
        rles = mu.frPyObjects(seg, h, w) if isinstance(seg, list) else [seg]
        rle = mu.merge(rles) if isinstance(rles, list) else rles
        c = np.asarray(_rle_counts(rle["counts"]), dtype=np.int64)
        ends = np.cumsum(c)
        starts = ends - c
        for s, e in zip(starts[1::2], ends[1::2]):
            flat[s:e] = i
        cls.append(CLASS_ID.get(cat[a["category_id"]], 0))
    return flat.reshape(w, h).T.copy(), np.asarray(cls, dtype=np.int32)


def gt_products(cfg, wall, coco_json=None, log=print):
    key = (paths.test_mask_path(cfg, wall), cfg.kernel_radius, coco_json)
    if key in _GT_CACHE:
        return _GT_CACHE[key]
    kr = cfg.kernel_radius
    gt_mask = gm._load_class_mask(paths.test_mask_path(cfg, wall))
    gt_bin = (gt_mask > 0).astype(np.uint8)
    roi, gt_gaps = gm.extract_gaps_via_closing(gt_bin, kr)
    stones = gm.extract_stones(gt_mask, MIN_STONE)
    unit_lab = np.zeros(gt_mask.shape, dtype=np.int32)
    unit_cls = [0]
    for i, s in enumerate(stones, start=1):
        unit_lab[s["slice"]][s["local_mask"]] = i
        unit_cls.append(s["class_id"])
    units = {"mask": (unit_lab, np.asarray(unit_cls, dtype=np.int32))}
    info = {"n_mask_units": len(stones)}
    if coco_json:
        try:
            plab, pcls = polygon_instance_raster(coco_json, cfg.ortho_pattern.format(wall=wall),
                                                 *gt_mask.shape)
            units["polygon"] = (plab, pcls)
            info["polygon_vs_mask_pixel_diff"] = int(np.count_nonzero((plab > 0) != (gt_mask > 0)))
            sizes = np.bincount(plab.ravel(), minlength=len(pcls))
            info["n_polygon_units"] = int(np.count_nonzero(sizes[1:] >= MIN_STONE))
        except Exception as e:  # polygons are an extra; never block the run
            log(f"[p2_eval] polygons for {wall} unavailable: {e!r}")
    prod = {"mask": gt_mask, "roi": roi.astype(bool), "gaps": gt_gaps, "stones": stones,
            "units": units, "info": info, "gt_stone_px": int(gt_bin.sum())}
    _GT_CACHE[key] = prod
    log(f"[GT] {wall}: {info}")
    return prod


# ---------------------------------------------------------------------------
# separation measure
# ---------------------------------------------------------------------------
def separation_status(unit_lab, unit_cls, pred_stone, conn, frac=FRAC, min_unit=MIN_STONE):
    """Per-GT-stone status table for one wall."""
    n_comp, lab = cv2.connectedComponents(pred_stone, connectivity=conn, ltype=cv2.CV_32S)
    n_units = len(unit_cls) - 1
    sizes = np.bincount(unit_lab.ravel(), minlength=n_units + 1)
    sel = (unit_lab > 0) & (lab > 0)
    base = np.int64(n_comp + 1)
    codes = unit_lab[sel].astype(np.int64) * base + lab[sel]
    del sel
    u, ov = np.unique(codes, return_counts=True)
    del codes
    uid, cid = u // base, u % base
    sig = ov >= frac * sizes[uid]
    uid_s, cid_s = uid[sig], cid[sig]
    comp_units = np.bincount(cid_s, minlength=int(base))
    n_sig = np.bincount(uid_s, minlength=n_units + 1)
    merged = np.zeros(n_units + 1, dtype=bool)
    merged[uid_s[comp_units[cid_s] >= 2]] = True
    ids = np.arange(1, n_units + 1)
    keep = sizes[1:] >= min_unit
    ids = ids[keep]
    df = pd.DataFrame({
        "unit_id": ids, "class_id": unit_cls[ids], "pixels": sizes[ids],
        "n_components": n_sig[ids],
        "undetected": n_sig[ids] == 0,
        "split": n_sig[ids] >= 2,
        "merged": merged[ids],
    })
    df["alone"] = (df["n_components"] == 1) & ~df["merged"]
    return df


def _summarise(df):
    n = len(df)
    out = {"n_stones": n}
    for k in ("alone", "merged", "split", "undetected"):
        c = int(df[k].sum())
        out[k] = c
        out[f"{k}_rate"] = c / n if n else np.nan
    return out


def _sep_rows(df, base):
    rows = [{**base, "class": "alle", **_summarise(df)}]
    for cname in ("Ashlar", "Polygonal", "Quarry"):
        sub = df[df["class_id"] == CLASS_ID[cname]]
        if len(sub):
            rows.append({**base, "class": cname, **_summarise(sub)})
    return rows


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
def eval_run(cfg, out_dir, coco_json=None, do_roi=True, force=False, log=print,
             save_details=True):
    """Evaluate one run (all cfg.walls). Returns (gapstone_df, separation_df)."""
    rid = make_run_id(cfg)
    exp = cfg.experiment_name
    os.makedirs(out_dir, exist_ok=True)
    gs_csv = os.path.join(out_dir, f"{rid}_gapstone.csv")
    sep_csv = os.path.join(out_dir, f"{rid}_separation.csv")

    if do_roi:
        _, rows = evaluate(cfg, force=force)
        append_to_manifest(cfg, rows)

    if os.path.exists(gs_csv) and os.path.exists(sep_csv) and not force:
        log(f"[skip-if-exists] p2 eval present: {exp} {rid}")
        return pd.read_csv(gs_csv), pd.read_csv(sep_csv)

    kr = cfg.kernel_radius
    gs_rows, sep_rows, all_evals = [], [], []
    counts = np.zeros(3, dtype=np.int64)
    area = np.zeros(2, dtype=np.int64)
    pooled = {}
    for wall in cfg.walls:
        raw = paths.segmentation_raw_path(cfg, wall)
        if not os.path.exists(raw):
            log(f"(missing) {exp} {rid} {wall} — skipped")
            continue
        g = gt_products(cfg, wall, coco_json, log)
        pred = gm._load_class_mask(raw)
        if pred.shape != g["mask"].shape:
            raise ValueError(f"{exp} {rid} {wall}: shape {pred.shape} vs GT {g['mask'].shape}")
        pred_bin = (pred > 0).astype(np.uint8)

        _, pred_gaps = gm.extract_gaps_via_closing(pred_bin, kr, g["roi"])
        tp, fp, fn = gm.gap_pixel_counts(g["gaps"], pred_gaps)
        del pred_gaps
        counts += (tp, fp, fn)
        labeled, info = gm.extract_prediction_regions(pred)
        evals = [gm.evaluate_stone(s, pred, labeled, info) for s in g["stones"]]
        del labeled
        all_evals.extend(evals)
        pred_in_roi = int(np.count_nonzero(pred_bin.astype(bool) & g["roi"]))
        area += (pred_in_roi, g["gt_stone_px"])
        gs_rows.append({"experiment": exp, "run_id": rid, "wall": wall, "kernel_radius": kr,
                        **gm.gap_metrics_from_counts(tp, fp, fn), **gm.stone_summary(evals),
                        "stone_area_ratio": pred_in_roi / g["gt_stone_px"]})

        for unit, (ulab, ucls) in g["units"].items():
            for conn in CONNS:
                df = separation_status(ulab, ucls, pred_bin, conn)
                base = {"experiment": exp, "run_id": rid, "wall": wall, "unit": unit,
                        "connectivity": conn}
                sep_rows.extend(_sep_rows(df, base))
                pooled.setdefault((unit, conn), []).append(df)
                if save_details:
                    det = os.path.join(out_dir, "details")
                    os.makedirs(det, exist_ok=True)
                    df.to_csv(os.path.join(det, f"{rid}_{wall}_{unit}_c{conn}.csv"), index=False)
        log(f"[ok] {exp} {rid} {wall}: gap IoU {gs_rows[-1]['gap_iou']:.3f} | "
            f"area ratio {gs_rows[-1]['stone_area_ratio']:.3f}")

    if gs_rows:
        gs_rows.append({"experiment": exp, "run_id": rid, "wall": "AllWalls", "kernel_radius": kr,
                        **gm.gap_metrics_from_counts(*counts), **gm.stone_summary(all_evals),
                        "stone_area_ratio": area[0] / area[1] if area[1] else np.nan})
        for (unit, conn), dfs in pooled.items():
            base = {"experiment": exp, "run_id": rid, "wall": "AllWalls", "unit": unit,
                    "connectivity": conn}
            sep_rows.extend(_sep_rows(pd.concat(dfs, ignore_index=True), base))
    gs_df, sep_df = pd.DataFrame(gs_rows), pd.DataFrame(sep_rows)
    n_done = len(gs_rows) - 1 if gs_rows else 0
    if n_done == len(cfg.walls):  # write only complete results (skip-if-exists keys on these files)
        gs_df.to_csv(gs_csv, index=False)
        sep_df.to_csv(sep_csv, index=False)
    else:
        log(f"[p2_eval] {exp} {rid}: only {n_done}/{len(cfg.walls)} walls — not written, retried later")
    return gs_df, sep_df
