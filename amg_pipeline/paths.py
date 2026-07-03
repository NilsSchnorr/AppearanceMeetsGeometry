"""
Path derivation. Given a RunConfig, every artifact location is computed from the
run_id so that runs are self-contained and never collide.

Layout under <experiments_root>/<experiment_name>/:
    checkpoints/<run_id>/model.pth
    checkpoints/<run_id>/config.json
    segmentations/<run_id>/<wall>_RAW_combined.png      (fed to the evaluator)
    segmentations/<run_id>/<wall>_segmented.png         (cleaned, for visual checks)
    metrics/<run_id>/<Wall>/roi_evaluation_<variant>_<op>.csv
    metrics/<run_id>/<Wall>/roi_evaluation_summary_<op>.csv
    manifest.csv                                        (one row per run x wall)
"""

import os
from .config import RunConfig, make_run_id


def experiment_dir(config: RunConfig) -> str:
    return os.path.join(config.experiments_root, config.experiment_name)


def checkpoint_dir(config: RunConfig) -> str:
    # Stage-0 stress test: checkpoints may be READ from another experiment
    # (checkpoint_experiment), while all outputs stay under experiment_name.
    # With the default "" this reproduces the original behavior exactly.
    exp = getattr(config, "checkpoint_experiment", "") or config.experiment_name
    return os.path.join(config.experiments_root, exp, "checkpoints", make_run_id(config))


def checkpoint_path(config: RunConfig) -> str:
    return os.path.join(checkpoint_dir(config), "model.pth")


def config_json_path(config: RunConfig) -> str:
    return os.path.join(checkpoint_dir(config), "config.json")


def segmentation_dir(config: RunConfig) -> str:
    return os.path.join(experiment_dir(config), "segmentations", make_run_id(config))


def segmentation_raw_path(config: RunConfig, wall: str) -> str:
    """The RAW (pre-cleanup) colored raster — this is what the evaluator reads."""
    return os.path.join(segmentation_dir(config), f"{wall}_RAW_combined.png")


def segmentation_clean_path(config: RunConfig, wall: str) -> str:
    """Cleaned grayscale raster — kept for visual inspection only."""
    return os.path.join(segmentation_dir(config), f"{wall}_segmented.png")


def metrics_dir(config: RunConfig) -> str:
    return os.path.join(experiment_dir(config), "metrics", make_run_id(config))


def metrics_wall_dir(config: RunConfig, wall: str) -> str:
    # "wall1" -> "Wall1" to match the existing 07_outputs_ROI-metrics layout
    pretty = wall.capitalize()
    return os.path.join(metrics_dir(config), pretty)


def manifest_path(config: RunConfig) -> str:
    return os.path.join(experiment_dir(config), "manifest.csv")


# ---- test-wall input paths (read-only inputs, not derived from run_id) -----
def test_ortho_path(config: RunConfig, wall: str) -> str:
    return os.path.join(config.test_ortho_dir, config.ortho_pattern.format(wall=wall))


def test_normalmap_path(config: RunConfig, wall: str) -> str:
    return os.path.join(config.test_normalmap_dir, config.normalmap_pattern.format(wall=wall))


def test_mask_path(config: RunConfig, wall: str) -> str:
    return os.path.join(config.test_mask_dir, config.mask_pattern.format(wall=wall))
