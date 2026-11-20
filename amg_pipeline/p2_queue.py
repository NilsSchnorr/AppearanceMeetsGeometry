"""
p2_queue.py — unattended queue for the P2 gap/joint experiments (Nov 2026).

Started by 05_orchestration/p2_fugen_orchestrator.ipynb as its own process:
    python -m amg_pipeline.p2_queue <queue_config.json>

Phases (all resume-safe: a restart only does what is still missing)
  0 checks       inputs present, free disk space
  1 morph check  fast exact closing == skimage on a real GT mask (and a paper
                 prediction); only then are the evaluators patched to use it
  2 reference    gap metrics + separation measure on the paper model's rasters
                 (7ch runs 1-5 and the ensemble); gap metrics compared with
                 v9_gapstone as a check of the evaluation path
  3 smoke test   every variant: 1 epoch on a few tiles, segment + evaluate the
                 smallest test wall; one 2-run ensemble; summary. Any error stops
                 the queue here, before the long runs.
  4 main queue   round-robin: run 1 of every variant, then run 2, ...
                 train -> segment -> evaluate. An error is logged and the queue
                 moves on; failed items are retried once at the end.
  5 ensembles    mean-softmax ensemble of all runs per variant, evaluated
  6 summary      p2_summary/ (also refreshed after every finished item)

Files in <experiments_root>/p2_summary/:
  status.json     phase, progress, ETA, failures (read by the notebook)
  heartbeat.txt   time stamp, refreshed every minute while the process lives
  queue_log.txt   everything the process prints (written by the notebook launcher)
  errors.log      tracebacks
  STOP            create this file to stop gracefully between two items
"""

import dataclasses
import gc
import glob
import json
import os
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta

import numpy as np

# Order = queue order within each round. Overrides are RunConfig fields.
VARIANTS = [
    ("p2_gapwins",         dict(mask_downsample="gap_wins")),
    ("p2_gapwins_bgw3",    dict(mask_downsample="gap_wins", bg_ce_weight=3.0)),
    ("p2_gapwins_bgw5",    dict(mask_downsample="gap_wins", bg_ce_weight=5.0)),
    ("p2_gapwins_wmap",    dict(mask_downsample="gap_wins", weight_map=True)),
    ("p2_gapwins_tversky", dict(mask_downsample="gap_wins", region_loss="tversky")),
    ("p2_gapwins_focal",   dict(mask_downsample="gap_wins", pixel_loss="focal")),
    ("p2_gapwins_cldice",  dict(mask_downsample="gap_wins", region_loss="dice+cldice")),
    ("p2_bgw3",            dict(bg_ce_weight=3.0)),
    ("p2_bgw5",            dict(bg_ce_weight=5.0)),
    ("p2_wmap",            dict(weight_map=True)),
    ("p2_tversky",         dict(region_loss="tversky")),
    ("p2_focal",           dict(pixel_loss="focal")),
    ("p2_cldice",          dict(region_loss="dice+cldice")),
]
PAPER_RUNS = "v2_yaw_correction-epsV1"
PAPER_ENS = "v2_baseline_ens"
ENS_SUFFIX = "_ens"
SUMMARY = "p2_summary"
MIN_FREE_GB = 15


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{_now()}] {msg}", flush=True)


class Status:
    def __init__(self, path):
        self.path = path
        self.d = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.d = json.load(f)
            except Exception:
                self.d = {}

    def set(self, **kw):
        self.d.update(kw)
        self.d["updated"] = _now()
        tmp = self.path + ".tmp"
        for attempt in range(5):  # Windows refuses os.replace while a reader holds the file
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.d, f, indent=2, ensure_ascii=False, default=str)
                os.replace(tmp, self.path)
                return
            except OSError:
                time.sleep(1 + attempt)
        log("(note) status.json could not be written this time; continuing")


def _heartbeat(path, stop_evt):
    while not stop_evt.is_set():
        try:
            with open(path, "w") as f:
                f.write(f"{time.time():.0f} {_now()} pid={os.getpid()}\n")
        except Exception:
            pass
        stop_evt.wait(60)


def _keep_awake(on=True):
    """Windows only: ask the OS not to sleep while the queue runs."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception as e:
        log(f"(note) keep-awake not available: {e!r}")


def _free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def _cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _err(sum_dir, what, exc):
    tb = traceback.format_exc()
    log(f"!!! ERROR in {what}: {exc!r}")
    with open(os.path.join(sum_dir, "errors.log"), "a", encoding="utf-8") as f:
        f.write(f"\n===== {_now()} | {what} =====\n{tb}\n")


# ---------------------------------------------------------------------------
def run(queue_cfg_path):
    with open(queue_cfg_path, encoding="utf-8") as f:
        q = json.load(f)

    from .config import RunConfig, make_run_id
    from . import paths
    from .train import train
    from .segment import segment
    from .ensemble import run_ensemble
    from .p2_eval import eval_run
    from .p2_summary import build_summary
    from . import p2_morph
    from . import gapmetrics as gm

    b = dict(q["base"])
    for k in ("model_size", "window_size"):  # JSON turns tuples into lists
        if isinstance(b.get(k), list):
            b[k] = tuple(b[k])
    base = RunConfig(**b)
    root = base.experiments_root
    sum_dir = os.path.join(root, SUMMARY)
    ref_dir = os.path.join(sum_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    st = Status(os.path.join(sum_dir, "status.json"))
    stop_file = os.path.join(sum_dir, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    n_runs = int(q.get("n_runs", 5))
    coco = q.get("coco_json") or None
    if coco and not os.path.exists(coco):
        log(f"(note) COCO JSON not found, polygon units skipped: {coco}")
        coco = None
    variants = [(n, ov) for n, ov in VARIANTS if not q.get("variants") or n in q["variants"]]
    names = [n for n, _ in variants]

    hb_stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(os.path.join(sum_dir, "heartbeat.txt"), hb_stop),
                     daemon=True).start()
    _keep_awake(True)
    st.set(phase="0_checks", pid=os.getpid(), started=st.d.get("started") or _now(),
           last_start=_now(), variants=names, n_runs=n_runs)

    def summary(note=""):
        try:
            build_summary(root, names, PAPER_RUNS, PAPER_ENS, n_runs, ref_dir, ENS_SUFFIX,
                          note=note)
        except Exception as e:
            _err(sum_dir, "summary", e)

    try:
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        log(f"python {sys.version.split()[0]} | torch {torch.__version__} | device {dev}")
    except Exception:
        pass

    # ---- 0 checks ---------------------------------------------------------
    problems, notes = [], []
    counts = {}
    for label, d in (("ortho", base.ortho_dir), ("normalmap", base.normalmap_dir), ("mask", base.mask_dir)):
        counts[label] = len(glob.glob(os.path.join(d, "*.png"))) if os.path.isdir(d) else 0
        if counts[label] == 0:
            problems.append(f"no training tiles in {label}: {d}")
    if len(set(counts.values())) > 1:
        problems.append(f"tile counts differ: {counts}")
    for wall in base.walls:
        for fn in (paths.test_ortho_path(base, wall), paths.test_normalmap_path(base, wall),
                   paths.test_mask_path(base, wall)):
            if not os.path.exists(fn):
                problems.append(f"missing test input: {fn}")
    for exp, rids in ((PAPER_RUNS, range(1, 6)), (PAPER_ENS, (1,))):
        for i in rids:
            c = dataclasses.replace(base, experiment_name=exp, run_number=i)
            for wall in base.walls:
                if not os.path.exists(paths.segmentation_raw_path(c, wall)):
                    notes.append(f"paper raster missing: {paths.segmentation_raw_path(c, wall)}")
    free = _free_gb(root)
    if free < MIN_FREE_GB:
        problems.append(f"only {free:.0f} GB free in {root}")
    log(f"[checks] tiles {counts} | free disk {free:.0f} GB | notes {len(notes)} | problems {len(problems)}")
    for n in notes:
        log("  (note) " + n)
    if problems:
        for p in problems:
            log("  !!! " + p)
        st.set(phase="checks_failed", problems=problems)
        hb_stop.set()
        _keep_awake(False)
        return 2
    st.set(checks={"tiles": counts, "free_gb": round(free), "notes": notes})

    smoke_wall = q.get("smoke_wall") or base.walls[0]
    if smoke_wall not in base.walls:
        smoke_wall = base.walls[0]

    # ---- 1 morph check ----------------------------------------------------
    st.set(phase="1_morph_check")
    try:
        from skimage.morphology import binary_closing as sk_closing
        gt = gm._load_class_mask(paths.test_mask_path(base, smoke_wall)) > 0
        ok1, nd1 = p2_morph.self_check(gt, base.kernel_radius, sk_closing)
        ok2, nd2 = True, 0
        pr = paths.segmentation_raw_path(dataclasses.replace(base, experiment_name=PAPER_ENS,
                                                             run_number=1), smoke_wall)
        if os.path.exists(pr):
            ok2, nd2 = p2_morph.self_check(gm._load_class_mask(pr) > 0, base.kernel_radius, sk_closing)
        if ok1 and ok2:
            p2_morph.patch_evaluators()
            log(f"[morph] fast closing identical on {smoke_wall} (GT and paper prediction) -> using it")
            st.set(morph="fast (identical)")
        else:
            log(f"[morph] fast closing differs ({nd1} / {nd2} px) -> keeping skimage (slow)")
            st.set(morph=f"skimage (fast differed: {nd1}/{nd2} px)")
        del gt
    except Exception as e:
        _err(sum_dir, "morph check", e)
        st.set(morph="skimage (check failed)")

    # ---- 2 reference ------------------------------------------------------
    if not q.get("skip_reference"):
        st.set(phase="2_reference")
        ref_check = {}
        v9 = os.path.join(root, "v9_gapstone", "gap_stone_metrics.csv")
        v9df = None
        if os.path.exists(v9):
            import pandas as pd
            v9df = pd.read_csv(v9)
        for exp, rids in ((PAPER_ENS, (1,)), (PAPER_RUNS, range(1, 6))):
            for i in rids:
                c = dataclasses.replace(base, experiment_name=exp, run_number=i)
                if not all(os.path.exists(paths.segmentation_raw_path(c, w)) for w in base.walls):
                    continue
                try:
                    gs, _ = eval_run(c, os.path.join(ref_dir, exp), coco, do_roi=False, log=log)
                    if v9df is not None and len(gs):
                        rid = make_run_id(c)
                        old = v9df[(v9df.experiment == exp) & (v9df.run_id == rid)].set_index("wall")
                        new = gs.set_index("wall")
                        common = [w for w in new.index if w in old.index]
                        if common:
                            diff = max(float(np.nanmax(np.abs(new.loc[common, k].astype(float).values
                                                               - old.loc[common, k].astype(float).values)))
                                       for k in ("gap_iou", "gap_tp", "gap_fp", "detection_rate_c90_s90"))
                            ref_check[f"{exp}/{rid}"] = diff
                            log(f"[reference] {exp} {rid}: max |new - v9_gapstone| = {diff:.3g}")
                except Exception as e:
                    _err(sum_dir, f"reference {exp} run{i}", e)
        st.set(reference_check=ref_check)
        summary()

    # ---- 3 smoke test -----------------------------------------------------
    if not q.get("skip_smoke"):
        st.set(phase="3_smoke_test")
        smoke_root = os.path.join(root, "p2_smoke")
        tiles = os.path.join(smoke_root, "_tiles")
        n_tiles = int(q.get("smoke_tiles", 32))
        t_smoke = time.time()
        try:
            dirs = {"ortho": base.ortho_dir, "normalmap": base.normalmap_dir, "mask": base.mask_dir}
            pick = sorted(os.path.basename(p) for p in glob.glob(os.path.join(base.mask_dir, "*.png")))[:n_tiles]
            prefixes = [p[:8] for p in pick]
            for label, d in dirs.items():
                dst = os.path.join(tiles, label)
                os.makedirs(dst, exist_ok=True)
                src = sorted(glob.glob(os.path.join(d, "*.png")))
                chosen = [s for s in src if os.path.basename(s)[:8] in prefixes]
                if len(chosen) != len(prefixes):
                    raise RuntimeError(f"smoke tiles: {label} has {len(chosen)} of {len(prefixes)} indices")
                for s in chosen:
                    t = os.path.join(dst, os.path.basename(s))
                    if not os.path.exists(t):
                        shutil.copy2(s, t)
            sbase = dataclasses.replace(base, experiments_root=smoke_root, n_epochs=1,
                                        walls=[smoke_wall],
                                        ortho_dir=os.path.join(tiles, "ortho"),
                                        normalmap_dir=os.path.join(tiles, "normalmap"),
                                        mask_dir=os.path.join(tiles, "mask"))
            times = {}
            for name, ov in variants:
                t0 = time.time()
                c = dataclasses.replace(sbase, experiment_name=name, run_number=1, **ov)
                log(f"[smoke] {name}")
                train(c, force=True)
                segment(c, force=True)
                eval_run(c, os.path.join(paths.experiment_dir(c), "p2_eval"), coco, do_roi=True,
                         force=True, log=log)
                times[name] = round(time.time() - t0, 1)
                _cleanup()
            name0, ov0 = variants[0]
            c2 = dataclasses.replace(sbase, experiment_name=name0, run_number=2, **ov0)
            train(c2, force=True)
            run_ensemble(sbase, source_experiment=name0, out_experiment=name0 + ENS_SUFFIX,
                         channel_variants=(7,), n_runs=2, force=True)
            ce = dataclasses.replace(sbase, experiment_name=name0 + ENS_SUFFIX, run_number=1)
            eval_run(ce, os.path.join(paths.experiment_dir(ce), "p2_eval"), coco, do_roi=False,
                     force=True, log=log)
            build_summary(smoke_root, names, PAPER_RUNS, PAPER_ENS, 1, ref_dir, ENS_SUFFIX,
                          note="VORABTEST (1 Epoche, wenige Kacheln) – Zahlen bedeutungslos.")
            _cleanup()
            log(f"[smoke] PASSED in {time.time() - t_smoke:.0f} s: {times}")
            st.set(smoke="passed", smoke_seconds=times)
        except Exception as e:
            _err(sum_dir, "smoke test", e)
            st.set(phase="smoke_failed", smoke=f"FAILED: {e!r}")
            hb_stop.set()
            _keep_awake(False)
            return 1

    # ---- 4 main queue -----------------------------------------------------
    items = [(r, n, ov) for r in range(1, n_runs + 1) for n, ov in variants]

    def cfg_of(r, n, ov):
        return dataclasses.replace(base, experiment_name=n, run_number=r, **ov)

    def is_done(c):
        rid = make_run_id(c)
        ev = os.path.join(paths.experiment_dir(c), "p2_eval")
        return (os.path.exists(paths.checkpoint_path(c))
                and all(os.path.exists(paths.segmentation_raw_path(c, w)) for w in c.walls)
                and os.path.exists(os.path.join(ev, f"{rid}_gapstone.csv"))
                and os.path.exists(os.path.join(ev, f"{rid}_separation.csv")))

    durations, failed = [], []

    def do_item(r, n, ov):
        c = cfg_of(r, n, ov)
        t0 = time.time()
        trained = not os.path.exists(paths.checkpoint_path(c))
        train(c)
        segment(c)
        eval_run(c, os.path.join(paths.experiment_dir(c), "p2_eval"), coco, do_roi=True, log=log)
        if trained:
            durations.append(time.time() - t0)

    def progress():
        done = sum(is_done(cfg_of(*it)) for it in items)
        rest = len(items) - done
        eta = None
        if durations:
            eta = (datetime.now() + timedelta(seconds=rest * float(np.mean(durations)))).strftime("%Y-%m-%d %H:%M")
        return done, rest, eta

    def main_pass(todo, label):
        for r, n, ov in todo:
            if os.path.exists(stop_file):
                log("[queue] STOP file found -> stopping gracefully")
                st.set(phase="stopped")
                return False
            c = cfg_of(r, n, ov)
            if is_done(c):
                continue
            free = _free_gb(root)
            if free < MIN_FREE_GB:
                log(f"[queue] only {free:.0f} GB free -> stopping")
                st.set(phase="stopped_disk_full")
                return False
            st.set(phase=label, current=f"{n} run{r}", current_since=_now())
            log(f"===== {label}: {n} run{r} =====")
            try:
                do_item(r, n, ov)
                if (r, n) in failed:
                    failed.remove((r, n))
            except Exception as e:
                _err(sum_dir, f"{n} run{r}", e)
                if (r, n) not in failed:
                    failed.append((r, n))
            finally:
                _cleanup()
            done, rest, eta = progress()
            st.set(items_total=len(items), items_done=done, items_failed=[f"{a} run{b}" for b, a in failed],
                   eta=eta, mean_hours_per_run=round(float(np.mean(durations)) / 3600, 2) if durations else None)
            summary()
        return True

    if not main_pass(items, "4_main_queue"):
        hb_stop.set(); _keep_awake(False)
        return 0
    if failed:
        log(f"[queue] retrying {len(failed)} failed item(s) once")
        retry = [(r, n, dict(VARIANTS)[n]) for r, n in list(failed)]
        if not main_pass(retry, "4b_retry"):
            hb_stop.set(); _keep_awake(False)
            return 0

    # ---- 5 ensembles ------------------------------------------------------
    st.set(phase="5_ensembles", current=None)
    for n, ov in variants:
        if os.path.exists(stop_file):
            st.set(phase="stopped")
            break
        members = [cfg_of(r, n, ov) for r in range(1, n_runs + 1)]
        if not all(os.path.exists(paths.checkpoint_path(m)) for m in members):
            log(f"[ensemble] {n}: not all {n_runs} runs present -> skipped")
            continue
        try:
            st.set(current=f"{n}{ENS_SUFFIX}", current_since=_now())
            run_ensemble(base, source_experiment=n, out_experiment=n + ENS_SUFFIX,
                         channel_variants=(7,), n_runs=n_runs)
            ce = dataclasses.replace(base, experiment_name=n + ENS_SUFFIX, run_number=1)
            eval_run(ce, os.path.join(paths.experiment_dir(ce), "p2_eval"), coco, do_roi=False, log=log)
        except Exception as e:
            _err(sum_dir, f"ensemble {n}", e)
        finally:
            _cleanup()
        summary()

    # ---- 6 summary ----------------------------------------------------------
    summary()
    done, rest, _ = progress()
    st.set(phase="6_done", current=None, finished=_now(), items_done=done,
           items_failed=[f"{a} run{b}" for b, a in failed])
    log(f"[queue] finished: {done}/{len(items)} runs, failed: {failed}")
    hb_stop.set()
    _keep_awake(False)
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
