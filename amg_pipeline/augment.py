"""
On-the-fly photometric augmentation for training (Stage 1, Candidate 2).

Motivation (from the Stage-0 stress test): the models train on tiles whose
lighting statistics never vary, so appearance features are learned as if
absolute brightness/contrast were reliable class evidence. Under photometric
perturbation at test time the 4ch appearance model collapses, and even the 7ch
fusion model does not fully fall back on its geometry channels — it never saw
degraded RGB paired with intact normals during training. This module makes the
color signal unreliable-by-design during training: every epoch, every training
tile receives a freshly drawn random photometric perturbation on its COLOR
channels only. Alpha, normal channels, and masks are never touched, so the
geometry channels remain the consistent signal and the 3ch variant is
unaffected by construction.

Scope: TRAINING ONLY. train.py wraps only the train split (validation stays
clean and comparable to baseline runs); segmentation/evaluation never see this
module. Architecture is untouched — old and new checkpoints are
interchangeable in structure; provenance lives in config.json and the
checkpoint's "photo_aug" field.

Policy (fixed constants for the v7_photoaug screen, so the screen tests ONE
well-defined policy rather than a tuning space; ranges deliberately sit INSIDE
the Stage-0 stress extremes — train on plausible variation, probe beyond it):
  brightness  factor ~ U(0.70, 1.30)                      (stress grid: 0.60-1.40)
  contrast    factor ~ U(0.70, 1.30), pivot = mean color
              over wall pixels (alpha > 0)                (stress grid: 0.60-1.40)
  gamma       exp(U(-ln(4/3), +ln(4/3))) i.e. [0.75,1.33] (stress grid: 0.67-1.50)
  shadow ramp probability 0.30; linear multiplicative ramp
              1.0 -> k with k ~ U(0.50, 1.00), random
              direction (4 orientations)                  (stress grid: down to 0.30)

Conventions mirror perturb.py: all operations are channel-symmetric (identical
per color channel, so cv2's BGR memory order is irrelevant), math in float on
[0,1], final clamp to [0,1]. Draws use torch's global RNG, consistent with how
the pipeline already treats run-time stochasticity (unseeded DataLoader
shuffle): augmentation contributes to run-to-run variance, as augmentation
should.
"""

import math

import torch
from torch.utils.data import Dataset

# ---- policy constants (v7_photoaug screen) ----------------------------------
BRIGHT_RANGE = (0.70, 1.30)
CONTRAST_RANGE = (0.70, 1.30)
GAMMA_LOG_MAX = math.log(4.0 / 3.0)   # gamma in [0.75, 1.333], log-symmetric
SHADOW_PROB = 0.30
SHADOW_MIN = 0.50                      # ramp endpoint k ~ U(SHADOW_MIN, 1.0)

_SHADOW_DIRECTIONS = ("left", "right", "top", "bottom")


def _u(lo, hi):
    return lo + (hi - lo) * torch.rand(()).item()


def draw_params():
    """One random photometric draw. Separated from application for testability."""
    p = {
        "bright": _u(*BRIGHT_RANGE),
        "contrast": _u(*CONTRAST_RANGE),
        "gamma": math.exp(_u(-GAMMA_LOG_MAX, GAMMA_LOG_MAX)),
        "shadow_k": None,
        "shadow_dir": None,
    }
    if torch.rand(()).item() < SHADOW_PROB:
        p["shadow_k"] = _u(SHADOW_MIN, 1.0)
        p["shadow_dir"] = _SHADOW_DIRECTIONS[int(torch.randint(0, 4, ()).item())]
    return p


def apply_params(x, params, n_color=3, alpha_index=3):
    """
    Apply one photometric draw to the color slice of one CHW float tensor in
    [0,1]. Returns a NEW tensor; channels >= n_color (alpha, normals) are
    copied through untouched and the input is never modified.
    """
    color = x[:n_color].clone()

    # brightness (multiplicative)
    color = color * params["bright"]

    # contrast around the mean wall luminance (alpha > 0), like perturb.py
    if x.shape[0] > alpha_index:
        wall = x[alpha_index] > 0
        pivot = color[:, wall].mean() if bool(wall.any()) else color.mean()
    else:
        pivot = color.mean()
    color = pivot + (color - pivot) * params["contrast"]

    # gamma (needs non-negative input)
    color = color.clamp(0.0, 1.0) ** params["gamma"]

    # occasional linear shading ramp across the tile
    if params["shadow_k"] is not None:
        k, d = params["shadow_k"], params["shadow_dir"]
        h, w = color.shape[1], color.shape[2]
        if d in ("left", "right"):
            ramp = torch.linspace(1.0, k, w, dtype=color.dtype)
            if d == "left":            # dark side on the left
                ramp = ramp.flip(0)
            ramp = ramp.view(1, 1, w)
        else:
            ramp = torch.linspace(1.0, k, h, dtype=color.dtype)
            if d == "top":             # dark side on top
                ramp = ramp.flip(0)
            ramp = ramp.view(1, h, 1)
        color = color * ramp

    color = color.clamp(0.0, 1.0)
    if x.shape[0] > n_color:
        return torch.cat([color, x[n_color:]], dim=0)
    return color


def apply_photometric(x, n_color=3, alpha_index=3):
    """Random draw + application (the training-time entry point)."""
    return apply_params(x, draw_params(), n_color=n_color, alpha_index=alpha_index)


class PhotometricAugment(Dataset):
    """
    Dataset wrapper: per-sample, per-access photometric jitter on the color
    channels. Wrap ONLY the training split. The base dataset's tensors are
    never modified; every __getitem__ returns a freshly augmented copy, so
    each epoch sees a different lighting realization of every tile.
    """

    def __init__(self, base_dataset, n_color=3, alpha_index=3):
        self.base = base_dataset
        self.n_color = n_color
        self.alpha_index = alpha_index

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return apply_photometric(x, self.n_color, self.alpha_index), y
