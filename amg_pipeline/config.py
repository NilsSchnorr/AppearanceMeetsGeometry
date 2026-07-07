"""
Configuration schema for the AppearanceMeetsGeometry orchestration pipeline.

A single RunConfig dataclass carries everything one run needs: which variant,
which run number, the seed, all input paths, and all hyper-parameters. Every
output path is *derived* from the run_id (see paths.py), so nothing is ever
hard-coded and runs can never silently overwrite each other.

FIELD STATUS
------------
Fields are labelled by which review step first needs them:

  LIVE (Step 1, the tilt-corrected normal-map rerun):
    channels, run_number, seed, n_epochs, batch_size, lr,
    experiment_name, experiments_root, all dataset/test paths,
    model_size, window_size, stride, roi_operation, kernel_radius

  LIVE-with-reproducing-default (implemented now, defaulted to reproduce the
  original behavior exactly; later steps just change the value):
    width_mult           -> "base"  (Step 2 sweeps slim/base/wide)
    ce_weight            -> 0.5     (Step 3 shifts the CE/Dice ratio)
    use_weighted_sampler -> False   (Step 3: quarry patch oversampling)
    sampler_weight       -> 3.0     (Step 3: weight for quarry-containing tiles)
    train_fraction       -> 1.0     (Step 4 reduces training data 100/75/50/25%)
    checkpoint_experiment-> ""      (Stage 0 stress test: read checkpoints from
                                     another experiment; "" = this experiment,
                                     reproducing the original behavior)
    norm                 -> "none"  (Stage 1: normalization inside DoubleConv;
                                     "groupnorm" inserts GroupNorm(8) after each
                                     conv. "none" reproduces the original
                                     architecture byte-identically)
    photo_aug            -> "none"  (Stage 1: on-the-fly photometric jitter on
                                     the COLOR channels of training tiles only;
                                     "rgb" enables it (no-op for 3ch). "none"
                                     reproduces the original training exactly)

  RESERVED (key present + validated, logic intentionally NOT implemented yet;
  setting them raises a clear error so they can't be used by accident):
    eval_type            -> "roi" (Step 5 adds "gap"/"stone_detection"/"basic")
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import json
import os


@dataclass
class RunConfig:
    # ---- variant / run identity (LIVE) ------------------------------------
    channels: int                      # 3, 4, or 7
    run_number: int                    # 1..n_runs
    experiment_name: str               # e.g. "v2_yaw_corrected"
    experiments_root: str              # absolute path to the experiments/ root

    # ---- training data (LIVE) ---------------------------------------------
    # 3ch ignores ortho_dir; 4ch ignores normalmap_dir; 7ch uses both.
    ortho_dir: str = ""
    normalmap_dir: str = ""
    mask_dir: str = ""

    # ---- test walls (LIVE) ------------------------------------------------
    test_ortho_dir: str = ""
    test_normalmap_dir: str = ""
    test_mask_dir: str = ""
    walls: List[str] = field(default_factory=lambda: ["wall1", "wall2", "wall3", "wall4"])
    # Filenames for each wall are built from these patterns ({wall} is substituted).
    # Defaults match the existing repo naming (note the GT mask reuses the ortho name).
    ortho_pattern: str = "{wall}_png-ortho.png"
    normalmap_pattern: str = "{wall}_DEM_normalmap.png"
    mask_pattern: str = "{wall}_png-ortho.png"

    # ---- training hyper-parameters (LIVE) ---------------------------------
    seed: int = 42
    n_epochs: int = 300
    batch_size: int = 16
    lr: float = 1e-4
    img_size: int = 512

    # ---- segmentation params (LIVE) ---------------------------------------
    model_size: Tuple[int, int] = (512, 512)
    window_size: Tuple[int, int] = (1280, 1280)
    stride: int = 960

    # ---- ROI evaluation params (LIVE) -------------------------------------
    roi_operation: str = "closing"     # "closing" or "dilation"
    kernel_radius: int = 45

    # ---- LIVE-with-reproducing-default ------------------------------------
    width_mult: str = "base"           # Step 2
    ce_weight: float = 0.5             # Step 3; dice weight = 1 - ce_weight
    use_weighted_sampler: bool = False # Step 3: oversample quarry-containing tiles
    sampler_weight: float = 3.0        # Step 3: weight for quarry tiles (1.0 = off)
    train_fraction: float = 1.0        # Step 4
    checkpoint_experiment: str = ""    # Stage 0 stress: checkpoint source ("" = own experiment)
    norm: str = "none"                 # Stage 1: "none" or "groupnorm" (DoubleConv normalization)
    photo_aug: str = "none"            # Stage 1: "none" or "rgb" (train-time color jitter)

    # ---- RESERVED (not implemented yet) -----------------------------------
    eval_type: str = "roi"              # Step 5

    # -----------------------------------------------------------------------
    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.channels not in (3, 4, 7):
            raise ValueError(f"channels must be 3, 4, or 7; got {self.channels}")
        if self.run_number < 1:
            raise ValueError(f"run_number must be >= 1; got {self.run_number}")
        if self.width_mult not in ("slim", "base", "wide"):
            raise ValueError(f"width_mult must be slim/base/wide; got {self.width_mult!r}")
        if not (0.0 <= self.ce_weight <= 1.0):
            raise ValueError(f"ce_weight must be in [0,1]; got {self.ce_weight}")
        if not (0.0 < self.train_fraction <= 1.0):
            raise ValueError(f"train_fraction must be in (0,1]; got {self.train_fraction}")
        if self.roi_operation not in ("closing", "dilation"):
            raise ValueError(f"roi_operation must be closing/dilation; got {self.roi_operation!r}")
        if self.sampler_weight <= 0:
            raise ValueError(f"sampler_weight must be > 0; got {self.sampler_weight}")
        if self.norm not in ("none", "groupnorm"):
            raise ValueError(f"norm must be none/groupnorm; got {self.norm!r}")
        if self.photo_aug not in ("none", "rgb"):
            raise ValueError(f"photo_aug must be none/rgb; got {self.photo_aug!r}")
        # Reserved-feature guards: fail loudly rather than silently misbehave.
        if self.eval_type != "roi":
            raise NotImplementedError(
                f"eval_type={self.eval_type!r} is reserved for Step 5 (full statistics "
                "rerun). Only 'roi' is implemented. Leave it as 'roi' for now."
            )

    # ---- convenience ------------------------------------------------------
    @property
    def n_classes(self) -> int:
        return 4

    def to_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=list)

    @classmethod
    def from_json(cls, path: str) -> "RunConfig":
        with open(path) as f:
            d = json.load(f)
        # tuples come back as lists from JSON
        for k in ("model_size", "window_size"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return cls(**d)


def make_run_id(config: RunConfig) -> str:
    """
    Derive the parseable run_id from the config.

    Step 1 uses the brief's scheme: "{channels}ch_run{N}".

    Later steps add axes by extending this function (the components are kept
    explicit and ordered so the names stay readable and stable):
      Step 2: prepend width      -> "{width}_{channels}ch_run{N}"  (when != base)
      Step 3: (no run_id change) oversampling/loss variants are isolated by a
              distinct experiment_name; full provenance lives in config.json
      Step 4: append "_frac{pct}" (when train_fraction != 1.0)
      Stage 1: (no run_id change) recipe variants (norm, ...) are isolated by a
              distinct experiment_name, exactly like Step 3
    Only the non-default axes are encoded, so Step-1 ids stay clean.
    """
    parts = []
    if config.width_mult != "base":          # dormant until Step 2
        parts.append(config.width_mult)
    parts.append(f"{config.channels}ch")
    if config.train_fraction != 1.0:         # dormant until Step 4
        parts.append(f"frac{int(round(config.train_fraction * 100))}")
    parts.append(f"run{config.run_number}")
    return "_".join(parts)
