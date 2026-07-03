"""
Stage-0 lighting-robustness stress test: re-segment and re-evaluate the FROZEN
baseline checkpoints on photometrically perturbed test orthomosaics.

No training happens here. For each perturbation condition this driver
  1. writes perturbed copies of the four test orthos into
         <stress_inputs_root>/<condition>/            (skip-if-exists per file)
  2. builds per-run configs that
       - READ checkpoints from the frozen experiment (checkpoint_experiment),
       - READ orthos from the condition folder (test_ortho_dir),
       - WRITE segmentations/metrics/manifest under their own experiment
         "<stress_base>_<condition>"  (one experiment dir per condition,
         exactly the Step-4 per-fraction pattern),
  3. runs segment -> evaluate -> append_to_manifest via the existing pipeline.

The frozen experiment is only ever read, never written. The 'none' condition
uses the ORIGINAL test-ortho folder directly (no copy), so it must reproduce
the frozen baseline numbers — a built-in sanity check of the whole harness.

Normal maps and GT masks are shared, untouched inputs for every condition;
the 3ch geometry model is therefore invariant by construction and is not
re-run here (its reference comes from the frozen manifest).
"""

import dataclasses
import os

import pandas as pd

from . import paths
from .config import make_run_id
from .evaluate import append_to_manifest, evaluate
from .perturb import apply_perturbation_to_file, condition_name, parse_condition
from .segment import segment


def stress_experiment_name(stress_base, condition):
    return f"{stress_base}_{condition}"


def prepare_condition_inputs(base_config, stress_inputs_root, family, level, force=False):
    """
    Write perturbed copies of every test-wall ortho for one condition.
    Returns the condition's ortho directory. Deterministic + skip-if-exists,
    so the folder can be deleted after evaluation and regenerated on demand.
    """
    cond = condition_name(family, level)
    dst_dir = os.path.join(stress_inputs_root, cond)
    for wall in base_config.walls:
        src = paths.test_ortho_path(base_config, wall)
        dst = os.path.join(dst_dir, base_config.ortho_pattern.format(wall=wall))
        if os.path.exists(dst) and not force:
            print(f"[skip-if-exists] perturbed ortho present: {dst}")
            continue
        print(f"[perturb] {cond}: {os.path.basename(src)} -> {dst}")
        apply_perturbation_to_file(src, dst, family, level)
    return dst_dir


def run_stress(base_config, frozen_experiment, stress_base, conditions,
               channel_variants=(4, 7), n_runs=5, stress_inputs_root=None,
               do_segment=True, do_evaluate=True, force=False):
    """
    Execute the stress sweep.

    conditions: iterable of (family, level) tuples; include ("none", 1.0) to
                run the identity condition through the identical harness.
    frozen_experiment: experiment name whose checkpoints are evaluated
                (e.g. the frozen baseline). Only read, never written.
    stress_base: base experiment name; each condition gets
                "<stress_base>_<condition>" (own manifest, no collisions).
    stress_inputs_root: where perturbed ortho folders are written
                (default: <experiments_root>/<stress_base>_inputs).

    Every stage is skip-if-exists, so an interrupted sweep resumes cleanly.
    """
    conditions = list(conditions)
    if stress_inputs_root is None:
        stress_inputs_root = os.path.join(base_config.experiments_root,
                                          f"{stress_base}_inputs")

    total = len(conditions) * len(channel_variants) * n_runs
    print(f"=== STRESS SWEEP: {total} seg+eval passes "
          f"({len(conditions)} conditions x {len(channel_variants)} variants x {n_runs} runs) ===")
    print(f"    checkpoints from: {frozen_experiment} (read-only)")

    summary = []
    for family, level in conditions:
        cond = condition_name(family, level)
        if family == "none":
            ortho_dir = base_config.test_ortho_dir  # originals, no copy
        else:
            ortho_dir = prepare_condition_inputs(base_config, stress_inputs_root,
                                                 family, level, force=False)
        for ch in channel_variants:
            for run_n in range(1, n_runs + 1):
                cfg = dataclasses.replace(
                    base_config,
                    channels=ch, run_number=run_n,
                    experiment_name=stress_experiment_name(stress_base, cond),
                    checkpoint_experiment=frozen_experiment,
                    test_ortho_dir=ortho_dir,
                )
                rid = make_run_id(cfg)
                print(f"\n----- [{cond}] {rid} -----")
                ckpt = paths.checkpoint_path(cfg)
                if not os.path.exists(ckpt):
                    raise FileNotFoundError(
                        f"frozen checkpoint missing: {ckpt}\n"
                        f"(check FROZEN_EXPERIMENT — it must name the experiment "
                        f"folder that holds the baseline checkpoints)")
                if do_segment:
                    segment(cfg, force=force)
                if do_evaluate:
                    _, rows = evaluate(cfg, force=force)
                    for r in rows:  # enrich with condition columns for plotting
                        r["condition"] = cond
                        r["family"] = family
                        r["level"] = level
                    append_to_manifest(cfg, rows)
                    summary.append((cond, rid))
    print("\n=== STRESS SWEEP COMPLETE ===")
    return summary


def collect_stress_manifests(base_config, stress_base, conditions,
                             frozen_experiment=None):
    """
    Read every condition manifest into one DataFrame (adding condition/family/
    level columns where older rows lack them). If frozen_experiment is given,
    its manifest is appended with condition='frozen' — the untouched reference,
    including the 3ch rows that the stress sweep deliberately skips.
    """
    frames = []
    for family, level in conditions:
        cond = condition_name(family, level)
        cfg = dataclasses.replace(base_config,
                                  experiment_name=stress_experiment_name(stress_base, cond))
        mpath = paths.manifest_path(cfg)
        if not os.path.exists(mpath):
            print(f"(missing) no manifest yet for {cond}: {mpath}")
            continue
        df = pd.read_csv(mpath)
        if "condition" not in df.columns:
            df["condition"], df["family"], df["level"] = cond, family, level
        frames.append(df)

    if frozen_experiment is not None:
        cfg = dataclasses.replace(base_config, experiment_name=frozen_experiment)
        mpath = paths.manifest_path(cfg)
        if os.path.exists(mpath):
            df = pd.read_csv(mpath)
            df["condition"], df["family"], df["level"] = "frozen", "frozen", 1.0
            frames.append(df)
        else:
            print(f"(missing) frozen manifest not found: {mpath}")

    if not frames:
        raise FileNotFoundError("no stress manifests found — run the sweep first")
    return pd.concat(frames, ignore_index=True)
