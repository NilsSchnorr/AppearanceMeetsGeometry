"""
segment(config) — runs sliding-window inference for all test walls using the
run's trained checkpoint, and saves per-wall RAW colored rasters (the input the
ROI evaluator reads) plus cleaned grayscale rasters (for visual checks only).
Skip-if-exists keyed per wall on the RAW raster.

Behavior is a faithful extraction of 03_image_segmentation_with_trained_ML/*:
  window 1280, stride 960, model size 512, center-weighted probability merge,
  then a minimal morphological cleanup (cleanup affects only the *_segmented.png
  visual output; metrics use the RAW raster, matching the paper's pipeline).
"""

import os
import gc
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import MultiUNet
from .data import CLASS_COLORS_RGB
from . import paths

FALLBACK_TO_CPU = True


def _load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_classes = checkpoint.get("n_classes", 4)
    img_channels = checkpoint.get("img_channels", 7)
    width_mult = checkpoint.get("width_mult", "base")
    norm = checkpoint.get("norm", "none")  # old checkpoints predate the field -> original arch
    model = MultiUNet(n_channels=img_channels, n_classes=n_classes, width_mult=width_mult,
                      norm=norm)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, n_classes, img_channels


def create_sliding_windows(image, window_size, stride):
    height, width = image.shape[:2]
    windows, positions = [], []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + window_size[0], height)
            x_end = min(x + window_size[1], width)
            y_start = max(0, y_end - window_size[0])
            x_start = max(0, x_end - window_size[1])
            window = image[y_start:y_end, x_start:x_end]
            if window.shape[0] != window_size[0] or window.shape[1] != window_size[1]:
                nw = np.zeros((window_size[0], window_size[1], image.shape[2]), dtype=image.dtype)
                nw[:window.shape[0], :window.shape[1]] = window
                window = nw
            windows.append(window)
            positions.append((y_start, x_start, y_end, x_end))
    return windows, positions

def _resize_multichannel(img, size, interpolation):
    """cv2.resize only accepts <=4 channels; for more (e.g. the 7ch stack),
    resize in <=4-channel blocks and re-concatenate. Numerically identical to
    resizing each component separately, since resize is per-channel."""
    c = img.shape[2] if img.ndim == 3 else 1
    if c <= 4:
        out = cv2.resize(img, size, interpolation=interpolation)
        return out[:, :, None] if out.ndim == 2 else out
    blocks = []
    for s in range(0, c, 4):
        r = cv2.resize(img[:, :, s:s+4], size, interpolation=interpolation)
        blocks.append(r[:, :, None] if r.ndim == 2 else r)
    return np.concatenate(blocks, axis=2)

def _segment_window(model, stacked_window, model_size, window_size, device):
    """stacked_window is already the correct channel stack (3/4/7) in HWC uint8."""
    try:
        resized = _resize_multichannel(stacked_window, model_size, cv2.INTER_AREA)
        norm = resized.astype("float32") / 255.0
        tensor = torch.FloatTensor(norm).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            probs_np = probs[0].cpu().numpy()
        if device == "cuda":
            torch.cuda.empty_cache()
        n_classes = probs_np.shape[0]
        probs_resized = np.zeros((n_classes, window_size[0], window_size[1]), dtype=np.float32)
        for c in range(n_classes):
            probs_resized[c] = cv2.resize(probs_np[c], (window_size[1], window_size[0]),
                                          interpolation=cv2.INTER_LINEAR)
        pred = np.argmax(probs_resized, axis=0).astype(np.uint8)
        return pred, probs_resized
    except RuntimeError as e:
        if "out of memory" in str(e) and device == "cuda" and FALLBACK_TO_CPU:
            torch.cuda.empty_cache(); gc.collect()
            model = model.to("cpu")
            return _segment_window(model, stacked_window, model_size, "cpu")
        raise


def combine_center_weighted(windows_data, positions, original_shape, n_classes, window_size):
    height, width = original_shape[:2]
    class_scores = np.zeros((height, width, n_classes), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    h, w = window_size
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    win_w = 1.0 - (dist / max_dist) * 0.7
    for (pred, probs), (y0, x0, y1, x1) in zip(windows_data, positions):
        ah, aw = y1 - y0, x1 - x0
        for c in range(n_classes):
            class_scores[y0:y1, x0:x1, c] += probs[c, :ah, :aw] * win_w[:ah, :aw]
        weights[y0:y1, x0:x1] += win_w[:ah, :aw]
    weights = np.maximum(weights, 1e-6)
    class_scores /= np.expand_dims(weights, axis=2)
    return np.argmax(class_scores, axis=2).astype(np.uint8)


def simple_cleanup(segmentation):
    cleaned = cv2.medianBlur(segmentation.astype(np.uint8), 3)
    kernel = np.ones((3, 3), np.uint8)
    for cid in range(1, 4):
        mask = (cleaned == cid).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned[cleaned == cid] = 0
        cleaned[mask == 1] = cid
    return cleaned


def _build_wall_stack(config, wall):
    """Load and stack the per-wall input into HWC according to channels."""
    ch = config.channels
    if ch in (4, 7):
        ortho = cv2.imread(paths.test_ortho_path(config, wall), cv2.IMREAD_UNCHANGED)
        if ortho is None:
            raise FileNotFoundError(paths.test_ortho_path(config, wall))
        if ortho.shape[2] != 4:
            alpha = np.full((ortho.shape[0], ortho.shape[1], 1), 255, dtype=ortho.dtype)
            ortho = np.concatenate([ortho, alpha], axis=2)
    if ch in (3, 7):
        normals = cv2.imread(paths.test_normalmap_path(config, wall), cv2.IMREAD_COLOR)
        if normals is None:
            raise FileNotFoundError(paths.test_normalmap_path(config, wall))

    if ch == 7:
        rgb, alpha = ortho[:, :, :3], ortho[:, :, 3:]
        return np.concatenate([rgb, alpha, normals], axis=2)
    if ch == 4:
        return ortho
    return normals  # ch == 3


def segment(config, force=False):
    """Segment all walls for this run. Returns a per-wall status dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(paths.segmentation_dir(config), exist_ok=True)

    ckpt = paths.checkpoint_path(config)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint missing, train first: {ckpt}")

    model = n_classes = None
    colors = np.array(CLASS_COLORS_RGB)
    results = {}

    for wall in config.walls:
        raw_path = paths.segmentation_raw_path(config, wall)
        if os.path.exists(raw_path) and not force:
            print(f"[skip-if-exists] segmentation present: {raw_path}")
            results[wall] = "skipped"
            continue

        if model is None:  # lazy-load the model only when work remains
            model, n_classes, ch = _load_model(ckpt, device)
            print(f"=== segment {paths.segmentation_dir(config)} | {ch}ch | device={device} ===")

        stack = _build_wall_stack(config, wall)
        windows, positions = create_sliding_windows(stack, config.window_size, config.stride)
        windows_data = []
        for i, win in enumerate(tqdm(windows, desc=f"{wall}: windows")):
            windows_data.append(_segment_window(model, win, config.model_size,
                                                 config.window_size, device))
            if i % 10 == 0 and device == "cuda":
                torch.cuda.empty_cache()

        seg = combine_center_weighted(windows_data, positions, stack.shape, n_classes,
                                      config.window_size)

        # RAW colored raster (metrics input) — RGB->BGR for cv2.imwrite
        colored = np.zeros((*seg.shape, 3), dtype=np.uint8)
        for i in range(n_classes):
            colored[seg == i] = colors[i]
        cv2.imwrite(raw_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

        # cleaned grayscale (visual only)
        cv2.imwrite(paths.segmentation_clean_path(config, wall), simple_cleanup(seg))
        print(f"saved {raw_path}")
        results[wall] = "segmented"

    return results
