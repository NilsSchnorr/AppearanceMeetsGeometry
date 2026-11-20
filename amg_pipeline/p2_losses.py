"""
p2_losses.py — loss variants for the P2 gap/joint experiments (Nov 2026).

The published recipe is train.CEDiceLoss (0.5 * CE + 0.5 * Dice, no weights)
and stays untouched; train.py uses P2Loss only when config.uses_p2_loss.

P2Loss = ce_weight * pixel_term + (1 - ce_weight) * region_term

pixel_term (per pixel, then a weighted mean sum(w * l) / sum(w)):
  "ce"     cross-entropy; with all weights 1 this equals nn.CrossEntropyLoss()
  "focal"  (1 - p_t)^gamma * CE          (Lin et al. 2017; no alpha factor)
  weights  w = class weight of the target (bg_ce_weight for Background, 1 for
           stones; the same normalisation as nn.CrossEntropyLoss(weight=...))
           times the border weight map if given (Ronneberger et al. 2015)

region_term (per tile and class, then the mean, as CEDiceLoss):
  "dice"         soft Dice, identical to CEDiceLoss
  "tversky"      1 - TP / (TP + a_c FP + b_c FN)   (Salehi et al. 2017)
                 stones: a = tversky_stone_fp (a stone predicted on a gap costs
                 more), b = 1 - a; background mirrored (a = 1 - s, b = s)
  "dice+cldice"  (1 - lambda) * Dice + lambda * clDice on the background
                 probability inside the tile ROI (Shit et al. 2021; soft
                 skeleton as in the reference implementation)
References are given as found, not re-checked here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- clDice (soft skeleton, reference implementation) ---------------------
def _soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def soft_skel(img, iters):
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(pred, true, iters=5, smooth=1.0):
    """pred, true: (B, 1, H, W) in [0, 1]."""
    sp = soft_skel(pred, iters)
    st = soft_skel(true, iters)
    tprec = ((sp * true).sum() + smooth) / (sp.sum() + smooth)
    tsens = ((st * pred).sum() + smooth) / (st.sum() + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


# ---- combined loss ---------------------------------------------------------
class P2Loss(nn.Module):
    def __init__(self, ce_weight=0.5, n_classes=4, bg_ce_weight=1.0,
                 pixel_loss="ce", focal_gamma=2.0, region_loss="dice",
                 tversky_stone_fp=0.7, cldice_lambda=0.5, cldice_iters=5,
                 smooth=1e-5):
        super().__init__()
        self.ce_w = ce_weight
        self.region_w = 1.0 - ce_weight
        self.pixel_loss = pixel_loss
        self.gamma = focal_gamma
        self.region_loss = region_loss
        self.cl_lambda = cldice_lambda
        self.cl_iters = cldice_iters
        self.smooth = smooth
        cw = torch.ones(n_classes)
        cw[0] = bg_ce_weight
        self.register_buffer("class_w", cw)
        self.use_class_w = bg_ce_weight != 1.0
        a = torch.full((n_classes,), float(tversky_stone_fp))
        a[0] = 1.0 - float(tversky_stone_fp)
        self.register_buffer("tv_alpha", a)       # weight of FP per class
        self.register_buffer("tv_beta", 1.0 - a)  # weight of FN per class

    def forward(self, logits, targets, wmap=None, roi=None):
        logp = F.log_softmax(logits, dim=1)
        nll = F.nll_loss(logp, targets, reduction="none")          # (B, H, W)
        if self.pixel_loss == "focal":
            pt = torch.exp(-nll)
            pix = (1.0 - pt).pow(self.gamma) * nll
        else:
            pix = nll
        w = None
        if self.use_class_w:
            w = self.class_w[targets]
        if wmap is not None:
            w = wmap if w is None else w * wmap
        pixel_term = pix.mean() if w is None else (pix * w).sum() / w.sum()

        soft = logp.exp()
        onehot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (2, 3)
        tp = (soft * onehot).sum(dims)
        if self.region_loss == "tversky":
            fp = (soft * (1.0 - onehot)).sum(dims)
            fn = ((1.0 - soft) * onehot).sum(dims)
            ti = (tp + self.smooth) / (tp + self.tv_alpha * fp + self.tv_beta * fn + self.smooth)
            region = 1.0 - ti.mean()
        else:
            card = soft.sum(dims) + onehot.sum(dims)
            dice = (2.0 * tp + self.smooth) / (card + self.smooth)
            region = 1.0 - dice.mean()
            if self.region_loss == "dice+cldice":
                if roi is None:
                    raise ValueError("dice+cldice needs the tile ROI")
                r = roi.unsqueeze(1).float()
                pred_bg = soft[:, :1] * r
                true_bg = (targets == 0).unsqueeze(1).float() * r
                cl = soft_cldice(pred_bg, true_bg, self.cl_iters)
                region = (1.0 - self.cl_lambda) * region + self.cl_lambda * cl
        return self.ce_w * pixel_term + self.region_w * region


def build_p2_loss(config):
    return P2Loss(ce_weight=config.ce_weight, n_classes=config.n_classes,
                  bg_ce_weight=config.bg_ce_weight, pixel_loss=config.pixel_loss,
                  focal_gamma=config.focal_gamma, region_loss=config.region_loss,
                  tversky_stone_fp=config.tversky_stone_fp,
                  cldice_lambda=config.cldice_lambda, cldice_iters=config.cldice_iters)
