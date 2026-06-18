"""
amg_pipeline — source-of-truth pipeline for the AppearanceMeetsGeometry project.

Public API:
    RunConfig, make_run_id        (config.py)
    train(config)                 (train.py)
    segment(config)               (segment.py)
    evaluate(config)              (evaluate.py)
    run_sweep(...)                (sweep.py)
    verify_architecture(), verify_checkpoint_loads()  (verify.py)

The three original training/segmentation/evaluation notebooks were extracted
into this module (Option A) and moved to Archive/ for reference. This module is
the single thing that actually runs.
"""

from .config import RunConfig, make_run_id
from .train import train
from .segment import segment
from .evaluate import evaluate, append_to_manifest
from .sweep import run_sweep, build_configs
from .verify import verify_architecture, verify_checkpoint_loads

__all__ = [
    "RunConfig", "make_run_id", "train", "segment", "evaluate",
    "append_to_manifest", "run_sweep", "build_configs",
    "verify_architecture", "verify_checkpoint_loads",
]
