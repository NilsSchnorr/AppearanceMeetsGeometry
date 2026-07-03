"""
Photometric perturbations for the Stage-0 lighting-robustness stress test.

Purpose: simulate lighting variation on the TEST orthomosaics only, so the
frozen checkpoints can be re-evaluated under degraded appearance conditions.
Normal maps are never touched (they are separate files this module never
sees), and the alpha channel is preserved bit-for-bit — so the 3ch geometry
model is invariant to every condition here *by construction*, and the 4ch/7ch
comparison isolates how each model copes when appearance degrades.

Implementation notes:
  - All operations are channel-symmetric (identical for each color channel),
    so cv2's BGR memory order is irrelevant. Deliberately no hue/temperature
    shifts, to avoid any BGR/RGB convention risk.
  - Files are read with cv2.IMREAD_UNCHANGED and written back as PNG
    (lossless), exactly mirroring how segment.py reads them.
  - Math is done in float32 [0,1], then clipped and rounded back to uint8.
  - Everything is deterministic: same input + condition -> same output file,
    so perturbed condition folders can be deleted and regenerated at will.

Families and their level semantics:
  bright   : multiplicative brightness, out = x * level        (1.0 = identity)
  contrast : scale around the mean wall luminance,
             out = pivot + (x - pivot) * level                 (1.0 = identity)
             pivot = scalar mean over wall pixels (alpha > 0)
  gamma    : out = x ** level                                  (1.0 = identity)
  shadow   : horizontal multiplicative ramp from 1.0 (left edge)
             down to `level` (right edge), simulating raking
             light / partial shading                           (1.0 = identity)
  none     : identity (handled upstream by pointing at the original files;
             kept in the registry so condition lists can include it)
"""

import os

import cv2
import numpy as np

FAMILIES = ("none", "bright", "contrast", "gamma", "shadow")


def condition_name(family, level):
    """'bright', 0.6 -> 'bright060'; 'none' has no level."""
    if family == "none":
        return "none"
    if family not in FAMILIES:
        raise ValueError(f"unknown perturbation family {family!r}; expected one of {FAMILIES}")
    return f"{family}{int(round(level * 100)):03d}"


def parse_condition(condition):
    """Inverse of condition_name: 'bright060' -> ('bright', 0.6)."""
    if condition == "none":
        return "none", 1.0
    for fam in FAMILIES:
        if fam != "none" and condition.startswith(fam):
            return fam, int(condition[len(fam):]) / 100.0
    raise ValueError(f"cannot parse condition {condition!r}")


def apply_perturbation(color_float01, family, level, wall_mask=None):
    """
    Apply one perturbation to a HxWx3 float32 array in [0,1].
    wall_mask (HxW bool) selects wall pixels (alpha > 0); used only as the
    contrast pivot region. Returns a new array clipped to [0,1].
    """
    x = color_float01
    if family == "none" or level == 1.0:
        out = x.copy()  # every family is the identity at level 1.0
    elif family == "bright":
        out = x * level
    elif family == "contrast":
        region = x[wall_mask] if (wall_mask is not None and wall_mask.any()) else x
        pivot = float(region.mean())
        out = pivot + (x - pivot) * level
    elif family == "gamma":
        out = np.power(np.clip(x, 0.0, 1.0), level)
    elif family == "shadow":
        width = x.shape[1]
        ramp = np.linspace(1.0, level, width, dtype=np.float32)[None, :, None]
        out = x * ramp
    else:
        raise ValueError(f"unknown perturbation family {family!r}")
    return np.clip(out, 0.0, 1.0)


def apply_perturbation_to_file(src_path, dst_path, family, level):
    """
    Read one test ortho (RGBA or RGB PNG), perturb the color channels,
    preserve alpha untouched, and write a lossless PNG to dst_path.
    """
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(src_path)
    if img.dtype != np.uint8:
        raise ValueError(f"{src_path}: expected uint8 PNG, got dtype {img.dtype}")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"{src_path}: expected HxWx3 or HxWx4, got shape {img.shape}")

    has_alpha = img.shape[2] == 4
    color = img[:, :, :3]
    alpha = img[:, :, 3:] if has_alpha else None
    wall_mask = (alpha[:, :, 0] > 0) if has_alpha else None

    out01 = apply_perturbation(color.astype(np.float32) / 255.0, family, level, wall_mask)
    out8 = np.clip(np.rint(out01 * 255.0), 0, 255).astype(np.uint8)
    result = np.concatenate([out8, alpha], axis=2) if has_alpha else out8

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if not cv2.imwrite(dst_path, result):
        raise IOError(f"cv2.imwrite failed for {dst_path}")
    return dst_path
