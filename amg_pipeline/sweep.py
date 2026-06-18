"""
run_sweep — drives train -> segment -> evaluate for every (variant, run) and
appends results to the master manifest. Resume-safe: every stage skips work that
already exists, so an interrupted sweep continues cleanly on re-run.
"""

import dataclasses
from .config import RunConfig, make_run_id
from .train import train
from .segment import segment
from .evaluate import evaluate, append_to_manifest


def build_configs(base_config: RunConfig, channel_variants, n_runs):
    """Expand a base config into one RunConfig per (variant, run)."""
    configs = []
    for ch in channel_variants:
        for run_n in range(1, n_runs + 1):
            cfg = dataclasses.replace(base_config, channels=ch, run_number=run_n)
            configs.append(cfg)
    return configs


def run_sweep(base_config: RunConfig, channel_variants=(3, 4, 7), n_runs=5,
              do_train=True, do_segment=True, do_evaluate=True, force=False):
    """
    Execute the full sweep. Each stage is independently gated and skip-if-exists,
    so you can re-run safely or run stages separately.
    """
    configs = build_configs(base_config, channel_variants, n_runs)
    print(f"=== SWEEP: {len(configs)} runs "
          f"({len(channel_variants)} variants x {n_runs} runs) ===")
    summary = []
    for i, cfg in enumerate(configs, 1):
        rid = make_run_id(cfg)
        print(f"\n----- [{i}/{len(configs)}] {rid} -----")
        if do_train:
            train(cfg, force=force)
        if do_segment:
            segment(cfg, force=force)
        if do_evaluate:
            _, rows = evaluate(cfg, force=force)
            append_to_manifest(cfg, rows)
            summary.append((rid, rows))
    print("\n=== SWEEP COMPLETE ===")
    return summary
