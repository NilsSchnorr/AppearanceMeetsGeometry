"""
p2_morph.py — exact, fast binary closing with a disk footprint (P2, Nov 2026).

Why: the evaluation builds the ROI and the gap masks with
skimage.morphology.binary_closing(mask, disk(45)) over whole test walls. On
large walls that is minutes per call, and the P2 queue evaluates ~80 rasters.

How: closing = erosion(dilation(mask)). With a disk footprint of radius r
(skimage.disk(r) = all offsets with dx^2 + dy^2 <= r^2) both steps are
thresholds of an exact Euclidean distance transform:
  dilation: a pixel is set iff its distance to the nearest set pixel <= r
  erosion:  a pixel stays iff its distance to the nearest unset pixel  > r
cv2.distanceTransform(DIST_L2, DIST_MASK_PRECISE) is exact (Felzenszwalb).
Border handling follows skimage: nothing outside the image is set for the
dilation, and nothing outside the image counts as unset for the erosion.

Safety: this module never replaces anything by itself. p2_queue calls
self_check() on a real GT mask first and only patches the evaluators if the
fast result is pixel-identical to skimage; otherwise the original slow path
stays in use. The paper code (evaluate.py, gapmetrics.py) is not edited.
"""

import numpy as np
import cv2
from skimage.morphology import disk

_EPS = 1e-4  # no integer d^2 lies within (r^2, r^2 + 2r*_EPS], so this is exact


def disk_radius(footprint):
    """Return r if footprint equals skimage disk(r), else None."""
    fp = np.asarray(footprint)
    if fp.ndim != 2 or fp.shape[0] != fp.shape[1] or fp.shape[0] % 2 != 1:
        return None
    r = (fp.shape[0] - 1) // 2
    if not np.array_equal(fp.astype(bool), disk(r).astype(bool)):
        return None
    return r


def fast_binary_closing_r(image, r):
    """Exact binary closing of a 2-D boolean image with disk(r)."""
    img = np.asarray(image).astype(bool)
    if r <= 0 or not img.any():
        return img.copy()
    # dilation: distance to nearest set pixel (zeros of ~img) <= r
    dist_fg = cv2.distanceTransform((~img).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dil = dist_fg <= r + _EPS
    del dist_fg
    if dil.all():
        return dil
    # erosion: distance to nearest unset pixel of dil > r
    dist_bg = cv2.distanceTransform(dil.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist_bg > r + _EPS


def make_patched_binary_closing(original):
    """Wrap skimage's binary_closing: disk footprints go the fast path."""
    def binary_closing(image, footprint=None, *args, **kwargs):
        r = disk_radius(footprint) if footprint is not None else None
        if r is None or args or kwargs or np.asarray(image).ndim != 2:
            return original(image, footprint, *args, **kwargs)
        return fast_binary_closing_r(image, r)
    binary_closing.__p2_fast__ = True
    return binary_closing


def self_check(mask_bool, r, original):
    """Compare fast vs original on a real mask. Returns (identical, n_diff)."""
    ref = np.asarray(original(mask_bool.astype(bool), disk(r))).astype(bool)
    fast = fast_binary_closing_r(mask_bool, r)
    n_diff = int(np.count_nonzero(ref != fast))
    return n_diff == 0, n_diff


def patch_evaluators():
    """Patch evaluate.py and gapmetrics.py in THIS process only."""
    import importlib
    # import the submodules explicitly: the package re-exports a function named
    # "evaluate", which would shadow the module of the same name
    _ev = importlib.import_module(__package__ + ".evaluate")
    _gm = importlib.import_module(__package__ + ".gapmetrics")
    for mod in (_ev, _gm):
        if not getattr(mod.binary_closing, "__p2_fast__", False):
            mod.binary_closing = make_patched_binary_closing(mod.binary_closing)
