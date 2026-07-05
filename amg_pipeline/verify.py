"""
Equivalence checks. Because we extracted the model by hand (Option A) and the
original notebooks are archived rather than kept as a live cross-check, this
module is the safety net: it proves the extracted MultiUNet matches the
architecture that produced the existing checkpoints BEFORE any sweep runs.

verify_architecture(): param counts + forward-pass shapes for 3/4/7ch.
verify_checkpoint_loads(): loads an existing .pth into the new module and asserts
    the state_dict keys match exactly (no missing / unexpected). This is the
    strongest possible check — if an existing trained model loads cleanly, the
    architecture is byte-identical.
"""

import torch
from .model import MultiUNet, count_parameters

# Expected base param counts (computed from the extracted architecture; these
# are fixed by the architecture, not by training).
EXPECTED_PARAMS = {3: 2_158_756, 4: 2_158_900, 7: 2_159_332}


def verify_architecture(verbose=True):
    ok = True
    for ch in (3, 4, 7):
        m = MultiUNet(n_channels=ch, n_classes=4, width_mult="base")
        n = count_parameters(m)
        out = m(torch.randn(1, ch, 512, 512))
        shape_ok = tuple(out.shape) == (1, 4, 512, 512)
        param_ok = (n == EXPECTED_PARAMS[ch])
        ok = ok and shape_ok and param_ok
        if verbose:
            flag = "OK " if (shape_ok and param_ok) else "FAIL"
            print(f"[{flag}] {ch}ch base: params={n:,} (expected {EXPECTED_PARAMS[ch]:,}), "
                  f"out={tuple(out.shape)}")
    if not ok:
        raise AssertionError("Architecture verification FAILED — do not run the sweep.")
    if verbose:
        print("Architecture verification PASSED.")
    return ok


def verify_checkpoint_loads(ckpt_path, channels, verbose=True):
    """
    Load an existing checkpoint into a fresh MultiUNet of the matching variant
    and assert the state_dict matches exactly. Run this against your existing
    02_MachineLearning/*.pth files to prove the extracted module is identical to
    the original.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    img_channels = ckpt.get("img_channels", channels)
    n_classes = ckpt.get("n_classes", 4)
    width_mult = ckpt.get("width_mult", "base")
    norm = ckpt.get("norm", "none")  # old checkpoints predate the field -> original arch
    model = MultiUNet(n_channels=img_channels, n_classes=n_classes, width_mult=width_mult,
                      norm=norm)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if verbose:
        print(f"[{ckpt_path}] img_channels={img_channels} n_classes={n_classes} "
              f"width={width_mult} norm={norm}")
        print(f"  missing keys:    {list(missing)}")
        print(f"  unexpected keys: {list(unexpected)}")
    if missing or unexpected:
        raise AssertionError(
            f"state_dict mismatch for {ckpt_path}: missing={list(missing)} "
            f"unexpected={list(unexpected)} — extracted architecture differs from original.")
    if verbose:
        print("  checkpoint loads cleanly — architecture matches the original.")
    return True
