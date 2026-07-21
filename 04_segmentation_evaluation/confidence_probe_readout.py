"""
confidence_probe_readout.py — R3 #7 pixel-readout figure (Fig. 11 ingredient).

Probes the ensemble probability stack (wall1_probs.npz, written by
amg_pipeline/confidence.py) at four points along the facade and renders, in
the style of the original Archive/03_.../probability_visualization_styled.py
mock-up:

    Row 1: the ensemble segmentation (argmax of the probabilities — identical
           to the saved v2_baseline_ens raster, agreement 0.99998) with
           numbered white probe markers;
    Row 2: one probability bar chart per probe point (class colors, winner
           outlined, percentages on the bars).

Probe selection (auto, overridable via --points):
    The wall is split into four equal x-bands ("along the facade"); one probe
    per band, numbered left to right. Candidates are stone-classified pixels
    (argmax != Background) eroded by EDGE_MARGIN px so every marker sits on a
    stone interior, never on background or a boundary sliver. Two bands yield
    a near-certain probe (highest confidence, preferring class variety), two
    an ambiguous one (confidence closest to TARGET_AMBIGUOUS) — echoing the
    Results text ("some points receive near-complete confidence ..., others
    reveal different levels of uncertainty").

Outputs (next to the npz):
    wall1_probe_readout.png   figure (300 dpi)
    wall1_probe_points.csv    probe coordinates + per-class probabilities

Usage (from the repo root, npz path is the default):
    python 04_segmentation_evaluation/confidence_probe_readout.py
    python ...readout.py --points "1200,2100 5300,1800 9800,2600 14500,2200"
    python ...readout.py --points "... ... ... 14500,2200,quarry"        # snap to class
    python ...readout.py --points "... ... ... 14500,2200,quarry@0.55"   # + conf target
"""

import argparse
import csv
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CLASS_NAMES = ["Background", "Ashlar", "Polygonal", "Quarry"]
CLASS_COLORS = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
BAR_COLORS = ["black", "blue", "red", "gold"]
BAR_LABELS = ["BG", "Ashlar", "Polygonal", "Quarry"]

N_PROBES = 4
EDGE_MARGIN = 15          # px erosion: probes sit on stone interiors
TARGET_AMBIGUOUS = 0.55   # confidence target for the two ambiguous probes
CERTAIN_BANDS = (0, 2)    # bands (0-based, left to right) with near-certain probes

_here = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NPZ = os.path.normpath(os.path.join(
    _here, "..", "experiments", "v2_baseline_ens", "confidence",
    "7ch_run1", "wall1_probs.npz"))


def _load_rgb_to_class_mask():
    """Import the canonical mask decoder straight from amg_pipeline/data.py,
    bypassing the package __init__ (which pulls in torch, not needed here)."""
    import importlib.util
    path = os.path.normpath(os.path.join(_here, "..", "amg_pipeline", "data.py"))
    spec = importlib.util.spec_from_file_location("_amg_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rgb_to_class_mask


def add_corner_label(ax, text, fontsize=11):
    """White-boxed corner label, as in paper Figs. 7-10 (from the archive script)."""
    ax.text(0.99, 0.95, text, transform=ax.transAxes, fontsize=fontsize,
            ha="right", va="top",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white",
                      edgecolor="none", alpha=0.9))


def pick_probes(seg, conf, gt_stone=None):
    """One probe per x-band on eroded stone interiors (optionally restricted
    to the GT-annotated facade); mix of certain and ambiguous picks,
    preferring class variety among the certain ones."""
    h, w = seg.shape
    stone = seg > 0
    if gt_stone is not None:
        stone = stone & gt_stone
    stone = stone.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * EDGE_MARGIN + 1, 2 * EDGE_MARGIN + 1))
    interior = cv2.erode(stone, kernel).astype(bool)

    points, used_classes = [], []
    for band in range(N_PROBES):
        x0, x1 = band * w // N_PROBES, (band + 1) * w // N_PROBES
        band_mask = np.zeros_like(interior)
        band_mask[:, x0:x1] = interior[:, x0:x1]
        ys, xs = np.nonzero(band_mask)
        if ys.size == 0:
            raise RuntimeError(f"band {band}: no stone-interior candidates "
                               f"(reduce EDGE_MARGIN? GT coverage?)")
        c = conf[ys, xs]
        if band in CERTAIN_BANDS:
            # near-certain: among the top 1 % most confident candidates,
            # prefer a class not already used by an earlier certain probe
            order = np.argsort(c)[::-1]
            top = order[:max(1, ys.size // 100)]
            idx = top[0]
            for j in top:
                if seg[ys[j], xs[j]] not in used_classes:
                    idx = j
                    break
        else:
            # ambiguous: confidence closest to the target
            idx = int(np.argmin(np.abs(c - TARGET_AMBIGUOUS)))
        y, x = int(ys[idx]), int(xs[idx])
        points.append((x, y))
        used_classes.append(int(seg[y, x]))
    return points


def snap_to_class(seg, target_class, x, y, gt_stone=None, conf=None,
                  conf_target=None):
    """Snap (x, y) to an interior pixel predicted as target_class (eroded by
    EDGE_MARGIN as in pick_probes; the margin is halved automatically while
    the eroded mask comes up empty — small Quarry stones can vanish under the
    full erosion).

    Without conf_target: the nearest such pixel. With conf_target (0-1): the
    pixel whose winning confidence is closest to the target, searched within
    a growing radius around the seed (nearest pixel among confidence ties) —
    the same argmin |conf - target| criterion pick_probes uses for its
    ambiguous probes, restricted to one class."""
    mask = seg == target_class
    if gt_stone is not None:
        mask = mask & gt_stone
    if not mask.any():
        raise SystemExit(f"snap: no pixels predicted as "
                         f"{CLASS_NAMES[target_class]}")
    margin = EDGE_MARGIN
    interior = np.zeros_like(mask)
    while margin >= 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        interior = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        if interior.any():
            break
        margin //= 2
    if not interior.any():
        interior, margin = mask, 0
    ys, xs = np.nonzero(interior)
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    if conf_target is None:
        i = int(np.argmin(d2))
        extra = ""
    else:
        r = 250
        within = d2 <= r * r
        while not within.any() and r < max(seg.shape):
            r *= 2
            within = d2 <= r * r
        if not within.any():
            within[:] = True
        diff = np.abs(conf[ys, xs].astype(np.float32) - conf_target)
        diff[~within] = np.inf
        i = int(np.lexsort((d2, np.round(diff, 3)))[0])
        extra = (f", conf {conf[ys[i], xs[i]] * 100:.0f}% "
                 f"(target {conf_target * 100:.0f}%, radius {r} px)")
    print(f"snapped ({x},{y}) -> ({int(xs[i])},{int(ys[i])}) "
          f"[{CLASS_NAMES[target_class]}{extra}, margin {margin} px, "
          f"distance {np.sqrt(d2[i]):.0f} px]")
    return int(xs[i]), int(ys[i])


def make_figure(seg, probs, points, out_png):
    h, w = seg.shape
    n_classes = probs.shape[2]
    plt.rcParams["font.family"] = "sans-serif"

    fig = plt.figure(figsize=(16, 4 + 16 * h / w), facecolor="white")
    gs = fig.add_gridspec(2, N_PROBES, height_ratios=[16 * h / w, 3.2],
                          hspace=0.25, wspace=0.3)

    # Row 1: segmentation with numbered probe markers
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(seg, cmap=ListedColormap(CLASS_COLORS), vmin=0,
              vmax=len(CLASS_NAMES) - 1, interpolation="nearest")
    ax.axis("off")
    add_corner_label(ax, "Full Model")
    r = max(h, w) // 60
    for i, (x, y) in enumerate(points):
        ax.add_patch(plt.Circle((x, y), radius=r, fill=True,
                                facecolor="white", edgecolor="black",
                                linewidth=2))
        ax.text(x, y, str(i + 1), ha="center", va="center",
                fontsize=11, fontweight="bold")

    # Row 2: per-probe probability bars
    for i, (x, y) in enumerate(points):
        axb = fig.add_subplot(gs[1, i])
        p = probs[y, x, :].astype(np.float64)
        winner = int(np.argmax(p))
        bars = axb.bar(range(n_classes), p * 100, color=BAR_COLORS[:n_classes],
                       edgecolor="black", linewidth=0.5)
        bars[winner].set_edgecolor("lime")
        bars[winner].set_linewidth(2.5)
        axb.set_ylim(0, 105)
        axb.set_xticks(range(n_classes))
        axb.set_xticklabels(BAR_LABELS[:n_classes], fontsize=9)
        if i == 0:
            axb.set_ylabel("Probability (%)", fontsize=10)
        for bar, prob in zip(bars, p):
            if prob > 0.03:
                axb.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 2, f"{prob * 100:.0f}%",
                         ha="center", va="bottom", fontsize=8)
        axb.spines["top"].set_visible(False)
        axb.spines["right"].set_visible(False)
        axb.set_title(f"Point {i + 1}: {CLASS_NAMES[winner]}",
                      fontsize=10, pad=6)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--points", default=None,
                    help='override auto-selection: "x,y x,y x,y x,y"; a pair '
                         'may carry a class suffix ("x,y,quarry") to snap to '
                         'the nearest interior pixel predicted as that class, '
                         'optionally with a confidence target '
                         '("x,y,quarry@0.55")')
    ap.add_argument("--gt-mask", default=None,
                    help="path to the wall's GT annotation PNG; restricts "
                         "probes to the annotated facade (recommended — "
                         "excludes out-of-masonry-bond structures)")
    args = ap.parse_args()

    probs = np.load(args.npz)["probs"]          # (H, W, C) float16
    conf = probs.max(axis=2).astype(np.float32)
    seg = probs.argmax(axis=2).astype(np.uint8)

    gt_stone = None
    if args.gt_mask:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        rgb_to_class_mask = _load_rgb_to_class_mask()
        gt_mask = rgb_to_class_mask(np.array(
            Image.open(args.gt_mask).convert("RGB")))
        if gt_mask.shape != seg.shape:
            h = min(gt_mask.shape[0], seg.shape[0])
            w = min(gt_mask.shape[1], seg.shape[1])
            print(f"WARNING: GT mask shape {gt_mask.shape} != raster shape "
                  f"{seg.shape}; using the common {h}x{w} region")
            aligned = np.zeros(seg.shape, dtype=bool)
            aligned[:h, :w] = gt_mask[:h, :w] > 0
            gt_stone = aligned
        else:
            gt_stone = gt_mask > 0
        print(f"probes restricted to the annotated facade ({args.gt_mask})")
    else:
        print("NOTE: no --gt-mask given; probes restricted only to predicted "
              "stones (may include out-of-masonry-bond structures)")

    if args.points:
        points = []
        for tok in args.points.split():
            parts = tok.split(",")
            if len(parts) not in (2, 3):
                raise SystemExit(f"--points token '{tok}' is not "
                                 f"'x,y' or 'x,y,class'")
            x, y = int(parts[0]), int(parts[1])
            if len(parts) == 3:
                lut = {n.lower(): i for i, n in enumerate(CLASS_NAMES)}
                name, _, tgt = parts[2].lower().partition("@")
                if name not in lut:
                    raise SystemExit(f"unknown class '{parts[2]}' in --points "
                                     f"(one of: {', '.join(CLASS_NAMES)})")
                conf_target = None
                if tgt:
                    try:
                        conf_target = float(tgt)
                    except ValueError:
                        raise SystemExit(f"bad confidence target '{tgt}' in "
                                         f"'{tok}' (use e.g. quarry@0.55)")
                    if conf_target > 1.0:
                        conf_target /= 100.0   # "quarry@55" == "quarry@0.55"
                    if not 0.0 < conf_target <= 1.0:
                        raise SystemExit(f"confidence target '{tgt}' out of "
                                         f"range (use 0-1 or 1-100)")
                x, y = snap_to_class(seg, lut[name], x, y, gt_stone,
                                     conf=conf, conf_target=conf_target)
            points.append((x, y))
        if len(points) != N_PROBES:
            raise SystemExit(f"--points needs {N_PROBES} x,y pairs")
        for x, y in points:
            if seg[y, x] == 0:
                raise SystemExit(f"point ({x},{y}) lies on Background — "
                                 f"probes must sit on stones")
            if gt_stone is not None and not gt_stone[y, x]:
                raise SystemExit(f"point ({x},{y}) lies outside the annotated "
                                 f"facade (GT mask)")
    else:
        points = pick_probes(seg, conf, gt_stone)

    out_dir = os.path.dirname(os.path.abspath(args.npz))
    stem = os.path.basename(args.npz).replace("_probs.npz", "")

    # console readout (old script's text-output format)
    print("=" * 70)
    for i, (x, y) in enumerate(points):
        p = probs[y, x, :].astype(np.float64)
        cells = " | ".join(f"{n}: {v * 100:.1f}%"
                           for n, v in zip(CLASS_NAMES, p))
        print(f"Point {i + 1}: ({x}, {y}) | {cells} | "
              f"-> {CLASS_NAMES[int(np.argmax(p))]}")
    print("=" * 70)

    csv_path = os.path.join(out_dir, f"{stem}_probe_points.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["point", "x", "y", "predicted"]
                    + [f"p_{n}" for n in CLASS_NAMES])
        for i, (x, y) in enumerate(points):
            p = probs[y, x, :].astype(np.float64)
            wr.writerow([i + 1, x, y, CLASS_NAMES[int(np.argmax(p))]]
                        + [f"{v:.4f}" for v in p])
    print(f"saved {csv_path}")

    make_figure(seg, probs, points, os.path.join(out_dir,
                                                 f"{stem}_probe_readout.png"))


if __name__ == "__main__":
    main()
