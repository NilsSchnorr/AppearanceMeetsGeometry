"""
confidence_heatmap.py — R3 #7 heatmap panel (Fig. 11 ingredient).

Renders the ensemble confidence raster (wall1_confidence.png, written by
amg_pipeline/confidence.py) as a figure-ready heatmap: chosen colormap,
stretched value range, horizontal colorbar, paper-style corner label.

The raw confidence PNG is a 16-bit data raster without a legend; this script
produces the presentation version. Because the confidence distribution is
heavily skewed (median 0.996), the default value range starts at 0.25 — the
theoretical minimum of a 4-class top-1 probability — so the uncertain ~26 %
tail gains visible contrast instead of drowning in white.

Usage (from the repo root; defaults find the canonical raster):
    python 04_segmentation_evaluation/confidence_heatmap.py
    python ...heatmap.py --cmap gray,RdYlGn --vmin 0.25
One output PNG per requested colormap, written next to the input:
    wall1_heatmap_<cmap>_vmin<val>.png
"""

import argparse
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_here = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONF = os.path.normpath(os.path.join(
    _here, "..", "experiments", "v2_baseline_ens", "confidence",
    "7ch_run1", "wall1_confidence.png"))


def add_corner_label(ax, text, fontsize=11):
    """White-boxed corner label, as in paper Figs. 7-10."""
    ax.text(0.99, 0.95, text, transform=ax.transAxes, fontsize=fontsize,
            ha="right", va="top",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white",
                      edgecolor="none", alpha=0.9))


def render(conf, cmap, vmin, vmax, out_png):
    h, w = conf.shape
    fig_w = 16.0
    fig_h = fig_w * h / w + 1.6          # image + room for the colorbar
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    im = ax.imshow(conf, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.axis("off")
    add_corner_label(ax, "Confidence")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        fraction=0.05, pad=0.03, shrink=0.6)
    cbar.set_label("Prediction confidence", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--conf", default=DEFAULT_CONF,
                    help="confidence raster (uint16 PNG)")
    ap.add_argument("--cmap", default="gray,RdYlGn",
                    help="comma-separated matplotlib colormap names; one "
                         "output per colormap")
    ap.add_argument("--vmin", type=float, default=0.25,
                    help="lower end of the color scale (0.25 = theoretical "
                         "4-class minimum)")
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--max-width", type=int, default=6000,
                    help="downsample the raster to this width for rendering "
                         "(keeps memory sane; 0 = full resolution)")
    args = ap.parse_args()

    conf16 = np.array(Image.open(args.conf))
    if conf16.dtype != np.uint16:
        raise SystemExit(f"expected uint16 confidence raster, got "
                         f"{conf16.dtype}: {args.conf}")
    conf = conf16.astype(np.float32) / 65535.0
    if args.max_width and conf.shape[1] > args.max_width:
        scale = args.max_width / conf.shape[1]
        conf = cv2.resize(conf, (args.max_width,
                                 max(1, round(conf.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
        print(f"rendering at {conf.shape[1]}x{conf.shape[0]} "
              f"(downsampled for display; use --max-width 0 for full res)")

    out_dir = os.path.dirname(os.path.abspath(args.conf))
    stem = os.path.basename(args.conf).replace("_confidence.png", "")
    for cmap in [c.strip() for c in args.cmap.split(",") if c.strip()]:
        tag = f"vmin{args.vmin:.2f}".replace(".", "")
        render(conf, cmap, args.vmin, args.vmax,
               os.path.join(out_dir, f"{stem}_heatmap_{cmap}_{tag}.png"))


if __name__ == "__main__":
    main()
