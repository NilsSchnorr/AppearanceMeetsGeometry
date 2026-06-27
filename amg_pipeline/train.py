"""
train(config) — trains one model variant for one run and saves the checkpoint
plus a config.json for provenance. Skip-if-exists keyed on the checkpoint path.

Behavior is a faithful extraction of the original training notebooks:
  - same CEDiceLoss (ce_weight from config; 0.5 reproduces the original 0.5/0.5)
  - same Adam(lr, betas=(0.9,0.999), eps=1e-7), grad clipping max_norm=1.0
  - same 90/10 train/val split with random_state=0 (kept FIXED so the split is
    not a source of run-to-run variance — see note below)
  - same equal-class-weights behavior (USE_EQUAL_WEIGHTS=True -> weight=None)

SEEDING (per the agreed plan):
  config.seed drives torch.manual_seed / np.random.seed and reproduces the
  original cudnn settings (deterministic=True alongside benchmark=True, exactly
  as in the originals — intentionally left as-is this round). The DataLoader
  shuffle is left unseeded, matching the originals. With the seed fixed at 42,
  run-to-run variance therefore comes from cuDNN benchmark kernel selection and
  the unseeded shuffle (i.e. run-time/implementation variance, init+split held
  fixed). Changing config.seed later turns this into a seed sweep with no code
  change.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm

from .model import MultiUNet, initialize_weights, count_parameters
from .data import (load_training_tiles, encode_masks, make_quarry_oversampling_weights,
                   CLASS_NAMES, CLASS_COLORS_RGB)
from . import paths


class CEDiceLoss(nn.Module):
    """0.5*CE + 0.5*Dice in the original; ce_weight makes the ratio configurable."""

    def __init__(self, ce_weight=0.5, weight=None, smooth=1e-5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.smooth = smooth
        self.ce_w = ce_weight
        self.dice_w = 1.0 - ce_weight

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        inputs_soft = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        dims = (2, 3)
        intersection = (inputs_soft * targets_one_hot).sum(dims)
        cardinality = inputs_soft.sum(dims) + targets_one_hot.sum(dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1. - dice_score.mean()
        return self.ce_w * ce_loss + self.dice_w * dice_loss


def calculate_iou(predictions, labels, n_classes):
    ious = []
    predictions = torch.argmax(predictions, dim=1)
    for cls in range(n_classes):
        pred_mask = (predictions == cls)
        true_mask = (labels == cls)
        intersection = (pred_mask & true_mask).float().sum()
        union = (pred_mask | true_mask).float().sum()
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        ious.append(iou.item() if isinstance(iou, torch.Tensor) else iou)
    return np.mean(ious)


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Reproduce the originals' cudnn block verbatim (left as-is this round).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train(config, force=False):
    """Train one run. Returns a small summary dict. Skips if checkpoint exists."""
    ckpt_path = paths.checkpoint_path(config)
    if os.path.exists(ckpt_path) and not force:
        print(f"[skip-if-exists] checkpoint present, skipping training: {ckpt_path}")
        return {"status": "skipped", "checkpoint": ckpt_path}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(config.seed)
    print(f"=== train {paths.checkpoint_dir(config)} | device={device} | seed={config.seed} ===")

    # ---- load + prepare data (verbatim behavior) --------------------------
    images, masks = load_training_tiles(config)
    masks_enc, _ = encode_masks(masks)
    images = images.astype("float32") / 255.0

    # train_fraction: deterministic subsample of training tiles (Step 4).
    # 1.0 reproduces the original (no subsampling).
    if config.train_fraction < 1.0:
        rng = np.random.RandomState(config.seed)
        n_keep = int(round(len(images) * config.train_fraction))
        keep = rng.choice(len(images), size=n_keep, replace=False)
        images, masks_enc = images[keep], masks_enc[keep]
        print(f"train_fraction={config.train_fraction}: using {n_keep}/{len(keep)} tiles")

    X_train, X_val, y_train, y_val = train_test_split(
        images, masks_enc, test_size=0.1, random_state=0  # split FIXED (see module docstring)
    )

    X_train_t = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    y_val_t = torch.LongTensor(y_val)

    n_classes = config.n_classes
    img_channels = X_train.shape[3]
    assert img_channels == config.channels, (
        f"loaded {img_channels} channels but config says {config.channels}")

    # ---- model / loss / optim --------------------------------------------
    model = MultiUNet(n_channels=img_channels, n_classes=n_classes,
                      width_mult=config.width_mult).to(device)
    initialize_weights(model)
    criterion = CEDiceLoss(ce_weight=config.ce_weight, weight=None)  # equal weights, as originals
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-7)
    print(f"params={count_parameters(model):,} | width={config.width_mult} | ce/dice={config.ce_weight}/{1-config.ce_weight}")

    train_ds = TensorDataset(X_train_t, y_train_t)
    if config.use_weighted_sampler:
        # Step 3 oversampling: tiles containing quarry (class 3) get sampler_weight,
        # all others 1.0; drawn WITH replacement. num_samples == #train tiles keeps
        # epoch length (and thus step count) identical to the base run. No explicit
        # generator is passed, so the sampler shares the global RNG seeded in
        # _set_seed -- exactly like the shuffle=True path it replaces.
        weights, n_pos, n_tot = make_quarry_oversampling_weights(
            y_train, config.sampler_weight)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights), replacement=True)
        exposure = (n_pos * config.sampler_weight) / (
            n_pos * config.sampler_weight + (n_tot - n_pos))
        print(f"[oversampling] quarry tiles {n_pos}/{n_tot} "
              f"(natural {n_pos / n_tot:.1%}) | weight={config.sampler_weight} "
              f"-> expected sampled exposure ~{exposure:.1%}")
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                  shuffle=True)  # unseeded, as originals
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                            batch_size=config.batch_size, shuffle=False)

    history = {k: [] for k in ("loss", "val_loss", "accuracy", "val_accuracy",
                               "iou_metric", "val_iou_metric")}

    print(f"Starting training at {datetime.now():%Y-%m-%d %H:%M:%S}")
    for epoch in range(config.n_epochs):
        model.train()
        tr_loss = tr_correct = tr_total = tr_iou = 0.0
        it = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.n_epochs}") if epoch % 10 == 0 else train_loader
        for data, target in it:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            tr_total += target.numel()
            tr_correct += (predicted == target).sum().item()
            with torch.no_grad():
                tr_iou += calculate_iou(output, target, n_classes)

        model.eval()
        v_loss = v_correct = v_total = v_iou = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                v_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                v_total += target.numel()
                v_correct += (predicted == target).sum().item()
                v_iou += calculate_iou(output, target, n_classes)

        history["loss"].append(tr_loss / len(train_loader))
        history["val_loss"].append(v_loss / len(val_loader))
        history["accuracy"].append(tr_correct / tr_total)
        history["val_accuracy"].append(v_correct / v_total)
        history["iou_metric"].append(tr_iou / len(train_loader))
        history["val_iou_metric"].append(v_iou / len(val_loader))

        if (epoch + 1) % 10 == 0:
            print(f"[{epoch+1}/{config.n_epochs}] "
                  f"loss {history['loss'][-1]:.4f}->{history['val_loss'][-1]:.4f} | "
                  f"iou {history['iou_metric'][-1]:.4f}->{history['val_iou_metric'][-1]:.4f}")

    # ---- save checkpoint + provenance ------------------------------------
    os.makedirs(paths.checkpoint_dir(config), exist_ok=True)
    torch.save({
        "epoch": config.n_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": history["loss"][-1],
        "history": history,
        "n_classes": n_classes,
        "class_names": CLASS_NAMES,
        "class_colors": CLASS_COLORS_RGB,
        "img_channels": img_channels,
        "width_mult": config.width_mult,
        "seed": config.seed,
        "run_number": config.run_number,
    }, ckpt_path)
    config.to_json(paths.config_json_path(config))
    print(f"saved checkpoint -> {ckpt_path}")
    print(f"saved config.json -> {paths.config_json_path(config)}")

    return {"status": "trained", "checkpoint": ckpt_path,
            "final_val_iou": history["val_iou_metric"][-1]}
