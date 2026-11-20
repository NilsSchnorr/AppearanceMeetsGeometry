"""
Data loading helpers, extracted verbatim (in behavior) from the training and
evaluation notebooks.

Training-tile loading mirrors the original notebooks exactly:
  - cv2.imread with IMREAD_UNCHANGED (RGBA) / IMREAD_COLOR (normals)
  - cv2.resize to img_size with INTER_AREA (images) / INTER_NEAREST (masks)
    (P2: config.mask_downsample="gap_wins" switches the masks to downsample_mask's
    "gap wins" rule; the default "nearest" is the published behavior)
  - 7ch  = concat([rgb, alpha, normals])  (channel order RGB, A, NxNyNz)
  - 4ch  = rgba
  - 3ch  = normals
  - LabelEncoder on raw mask values {0,29,76,225} -> {0,1,2,3}
  - images normalized to float32/255.0

The LabelEncoder maps ascending raw values to 0..3, i.e.
  0->Background, 29->Ashlar, 76->Polygonal, 225->Quarry — consistent with the
  RGB_TO_CLASS mapping used at evaluation time.
"""

import glob
import os
import cv2
import numpy as np


# RGB -> class index, from the evaluation notebook (used for GT and predictions).
RGB_TO_CLASS = {
    (0, 0, 0): 0,       # Black  -> Background
    (0, 0, 255): 1,     # Blue   -> Ashlar
    (255, 0, 0): 2,     # Red    -> Polygonal
    (255, 255, 0): 3,   # Yellow -> Quarry
}

CLASS_NAMES = ["Background", "Ashlar", "Polygonal", "Quarry"]
CLASS_COLORS_RGB = [[0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0]]


def _sorted_pngs(directory):
    return sorted(glob.glob(os.path.join(directory, "*.png")))


def load_training_tiles(config):
    """
    Load training tiles for the given channel variant.

    Returns (images, masks) as numpy arrays:
      images: (N, H, W, C) float32 in [0,1], C == config.channels
      masks:  (N, H, W)     int64 class indices in {0,1,2,3}
    """
    ch = config.channels
    size = (config.img_size, config.img_size)

    if ch in (4, 7):
        ortho_paths = _sorted_pngs(config.ortho_dir)
    if ch in (3, 7):
        normal_paths = _sorted_pngs(config.normalmap_dir)
    mask_paths = _sorted_pngs(config.mask_dir)

    if ch == 7 and len(ortho_paths) != len(normal_paths):
        print("WARNING: #orthomosaics != #normalmaps")

    images = []
    if ch == 7:
        for rgba_path, normals_path in zip(ortho_paths, normal_paths):
            img_rgba = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)
            if img_rgba is None or img_rgba.shape[-1] != 4:
                print(f"Error loading RGBA image: {rgba_path}")
                continue
            img_rgba = cv2.resize(img_rgba, size, interpolation=cv2.INTER_AREA)
            normals = cv2.imread(normals_path, cv2.IMREAD_COLOR)
            if normals is None or normals.shape[-1] != 3:
                print(f"Error loading normals image: {normals_path}")
                continue
            normals = cv2.resize(normals, size, interpolation=cv2.INTER_AREA)
            rgb = img_rgba[:, :, :3]
            alpha = img_rgba[:, :, 3:]
            images.append(np.concatenate([rgb, alpha, normals], axis=2))
    elif ch == 4:
        for rgba_path in ortho_paths:
            img_rgba = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)
            if img_rgba is None or img_rgba.shape[-1] != 4:
                print(f"Error loading RGBA image: {rgba_path}")
                continue
            img_rgba = cv2.resize(img_rgba, size, interpolation=cv2.INTER_AREA)
            images.append(img_rgba)
    elif ch == 3:
        for normals_path in normal_paths:
            normals = cv2.imread(normals_path, cv2.IMREAD_COLOR)
            if normals is None or normals.shape[-1] != 3:
                print(f"Error loading normals image: {normals_path}")
                continue
            normals = cv2.resize(normals, size, interpolation=cv2.INTER_AREA)
            images.append(normals)

    images = np.array(images)

    masks = []
    mode = getattr(config, "mask_downsample", "nearest")
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, 0)
        mask = downsample_mask(mask, size, mode)
        masks.append(mask)
    masks = np.array(masks)

    return images, masks


# ---------------------------------------------------------------------------
# P2 (Nov 2026): training-target and per-tile loss helpers
# ---------------------------------------------------------------------------
def downsample_mask(mask, size, mode="nearest"):
    """Downsample a raw mask (0 = background) to `size` (w, h).

    "nearest"  : cv2.INTER_NEAREST, exactly as the published training.
    "gap_wins" : same, but a target cell becomes background as soon as ANY
                 source pixel it touches (fully or partly) is background, so a
                 1-2 px joint can no longer vanish ("Fuge gewinnt").
                 INTER_AREA of the background indicator is > 0 exactly for
                 those cells (checked against the exact cell overlap).
    """
    small = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    if mode == "gap_wins" and mask.shape[:2] != (size[1], size[0]):
        bg = (mask == 0).astype(np.float32)
        any_bg = cv2.resize(bg, size, interpolation=cv2.INTER_AREA) > 1e-6
        small[any_bg] = 0
    elif mode not in ("nearest", "gap_wins"):
        raise ValueError(f"unknown mask_downsample {mode!r}")
    return small


def border_weight_map(mask_enc, w0=10.0, sigma=5.0):
    """U-Net border weight map (Ronneberger et al. 2015), border term only.

    w(x) = 1 + w0 * exp(-(d1 + d2)^2 / (2 sigma^2)) on BACKGROUND pixels, where
    d1, d2 are the distances to the nearest and second-nearest stone; stone
    pixels keep w = 1. No class-balancing term (the class weights the paper's
    term would add over-corrected in earlier runs). Stones = 4-connected
    components of the (already downsampled) target, so a joint that the target
    lost never gets a high weight. Distances are computed in a window of 4*sigma
    around each stone; beyond that the term is below w0 * e^-8 and is dropped.
    """
    mask_enc = np.asarray(mask_enc)
    h, w = mask_enc.shape
    out = np.ones((h, w), dtype=np.float32)
    stone = (mask_enc > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(stone, connectivity=4)
    if n <= 2 or w0 == 0:
        return out
    m = int(np.ceil(4 * sigma))
    d1 = np.full((h, w), np.inf, dtype=np.float32)
    d2 = np.full((h, w), np.inf, dtype=np.float32)
    for k in range(1, n):
        x, y, bw, bh, _ = stats[k]
        y0, y1 = max(0, y - m), min(h, y + bh + m)
        x0, x1 = max(0, x - m), min(w, x + bw + m)
        sub = (lab[y0:y1, x0:x1] != k).astype(np.uint8)
        dk = cv2.distanceTransform(sub, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        D1 = d1[y0:y1, x0:x1]
        D2 = d2[y0:y1, x0:x1]
        d2[y0:y1, x0:x1] = np.where(dk < D1, D1, np.minimum(D2, dk))
        d1[y0:y1, x0:x1] = np.minimum(D1, dk)
    bg = (mask_enc == 0) & np.isfinite(d2)
    s = d1[bg] + d2[bg]
    out[bg] = 1.0 + w0 * np.exp(-(s * s) / (2.0 * sigma * sigma))
    return out


def tile_roi(mask_enc, radius):
    """Per-tile ROI as in the evaluation: closing of the stone mask with disk(radius)."""
    from .p2_morph import fast_binary_closing_r
    return fast_binary_closing_r(np.asarray(mask_enc) > 0, int(radius))


def encode_masks(masks):
    """
    Map raw mask values to class indices via ascending sort, reproducing
    sklearn LabelEncoder behavior on the original {0,29,76,225} encoding,
    but without depending on which values happen to be present in a subset.
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    n, h, w = masks.shape
    flat = masks.reshape(-1)
    enc = le.fit_transform(flat)
    return enc.reshape(n, h, w), le


def make_quarry_oversampling_weights(masks_enc, sampler_weight, target_class=3):
    """
    Per-tile sampling weights for a WeightedRandomSampler (Step 3 oversampling).

    Tiles that contain at least one pixel of `target_class` (3 == Quarry, per the
    LabelEncoder ordering above) receive `sampler_weight`; every other tile
    receives 1.0. This is the binary patch-level oversampling scheme agreed for
    the class-imbalance revision: it raises the model's exposure to quarry-
    containing contexts without touching the data or the loss. A weight of 1.0
    reproduces uniform sampling exactly.

    Note the exposure is non-linear in the weight. With n_pos quarry tiles and
    n_neg others, the expected fraction of draws landing on a quarry tile is
    (n_pos * w) / (n_pos * w + n_neg), NOT w x the natural fraction. train.py
    logs the realized fraction at run time so it can be confirmed.

    Parameters
    ----------
    masks_enc : np.ndarray, shape (N, H, W), integer class indices in {0,1,2,3}.
    sampler_weight : float, weight for quarry-containing tiles (>0; >1 oversamples).
    target_class : int, the class whose presence triggers up-weighting (default 3).

    Returns
    -------
    weights : np.ndarray, shape (N,), float64 - one weight per tile.
    n_pos   : int - number of tiles containing target_class.
    n_total : int - total number of tiles.
    """
    present = (masks_enc == target_class).reshape(masks_enc.shape[0], -1).any(axis=1)
    weights = np.where(present, float(sampler_weight), 1.0).astype("float64")
    return weights, int(present.sum()), int(present.shape[0])


def rgb_to_class_mask(rgb_image, verbose=False):
    """
    Convert an RGB mask to class indices. Unmapped pixels (compression /
    anti-aliasing artifacts) are assigned to the nearest mapped color.
    Behavior matches the evaluation notebook.
    """
    height, width = rgb_image.shape[:2]
    class_mask = np.zeros((height, width), dtype=np.uint8)

    for rgb_tuple, class_idx in RGB_TO_CLASS.items():
        color_mask = np.all(rgb_image == rgb_tuple, axis=2)
        class_mask[color_mask] = class_idx

    mapped = np.zeros((height, width), dtype=bool)
    for rgb_tuple in RGB_TO_CLASS.keys():
        mapped |= np.all(rgb_image == rgb_tuple, axis=2)

    unmapped_count = int(np.sum(~mapped))
    if unmapped_count > 0:
        if verbose:
            print(f"    Note: {unmapped_count} unmapped pixels -> nearest class")
        idxs = np.where(~mapped)
        palette = np.array(list(RGB_TO_CLASS.keys()), dtype=float)
        classes = np.array(list(RGB_TO_CLASS.values()))
        pix = rgb_image[idxs].astype(float)              # (M, 3)
        d = np.linalg.norm(pix[:, None, :] - palette[None, :, :], axis=2)  # (M, K)
        nearest = classes[np.argmin(d, axis=1)]
        class_mask[idxs] = nearest.astype(np.uint8)

    return class_mask
