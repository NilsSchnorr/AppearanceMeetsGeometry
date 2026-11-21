"""
p2_queue.py — unattended queue for the P2 gap/joint experiments (Nov 2026).

Started by 05_orchestration/p2_fugen_orchestrator.ipynb as its own process:
    python -m amg_pipeline.p2_queue <queue_config.json>

Process model (since 21.11.2026)
  The queue process computes nothing itself. Every step runs in a child
  process of its own,
      python -m amg_pipeline.p2_queue --job <job> ... <queue_config.json>
  so a crash in native code ends only that child. (The first queue died on
  20.11.2026 with 0xC0000005, an access violation, at the start of the first
  weight-map run; such a crash takes the whole process down without a Python
  traceback.) For every failed child the queue logs the exit code, the reason
  and, from faulthandler, the Python lines where it crashed or hung, then
  moves on; failed runs are retried once at the end.
  Watchdog: a child is killed if its heartbeat is older than hb_stale_s
  (default 20 min) or it runs longer than its time limit (LIMITS_H).
  Success = the child wrote its final result file; this does not depend on
  how Windows reports the exit code of the Python launcher.

Phases (all resume-safe: a restart only does what is still missing)
  0 checks       inputs present, free disk space (queue process)
  1 morph check  fast exact closing == skimage on a real GT mask (and a paper
                 prediction); only then do the children patch the evaluators
  2 reference    gap metrics + separation measure on the paper model's rasters
                 (7ch runs 1-5 and the ensemble); gap metrics compared with
                 v9_gapstone as a check of the evaluation path
  3 smoke test   every variant: 1 epoch on a few tiles, segment + evaluate the
                 smallest test wall, one child per variant. A variant that fails
                 is left out of the main queue; the others go on. Then one
                 2-run ensemble, and the per-tile extras (weight map, clDice
                 ROI) on ALL training tiles; if an extra fails, its variants are
                 left out. Only if every variant fails does the queue stop.
  4 main queue   round-robin: run 1 of every variant, then run 2, ...
                 one child per run: train -> segment -> evaluate.
  5 ensembles    mean-softmax ensemble of all runs per variant, evaluated
  6 summary      p2_summary/ (also refreshed after every finished run)

Files in <experiments_root>/p2_summary/:
  status.json         phase, progress, ETA, failures (read by the notebook)
  heartbeat.txt       time stamp of the queue process, refreshed every minute
  child_heartbeat.txt the same for the running child
  queue_log.txt       everything queue and children print (opened by the notebook)
  errors.log          Python tracebacks, and for crashes the faulthandler output
  crashes/            faulthandler output per failed child
  STOP                create this file to stop gracefully between two steps
"""

import argparse
import dataclasses
import faulthandler
import gc
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
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

# Time limits per child (hours). A run (300 epochs + segmentation + evaluation)
# took 1.9 h on 20.11.2026; 6 h only catches a child that hangs.
LIMITS_H = {"morph": 1, "reference": 4, "smoke": 1, "smoke_ens": 1, "preflight": 1,
            "item": 6, "ensemble": 4, "summary": 0.5}
HB_STALE_S = 20 * 60

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Windows status codes worth naming in the log
_NTSTATUS = {
    0xC0000005: "access violation in native code",
    0xC0000409: "fail-fast abort inside a library",
    0xC0000374: "heap corruption",
    0xC00000FD: "stack overflow",
    0xC0000017: "out of memory",
    0xC000013A: "console closed / Ctrl+C",
    0x40010004: "terminated from outside",
}


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{_now()}] {msg}", flush=True)


def describe_rc(rc):
    """Readable exit code (Windows NTSTATUS in hex, POSIX signals by number)."""
    if rc is None:
        return "unknown"
    if rc < 0:
        return f"signal {-rc}"
    u = rc & 0xFFFFFFFF
    if u in _NTSTATUS:
        return f"0x{u:08X} ({_NTSTATUS[u]})"
    if u == 1:
        return "1 (Python error or killed)"
    return str(u) if u < 256 else f"0x{u:08X}"


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


def _rm(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# configuration shared by queue and children
def _load_q(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _base_config(q):
    from .config import RunConfig
    b = dict(q["base"])
    for k in ("model_size", "window_size"):  # JSON turns tuples into lists
        if isinstance(b.get(k), list):
            b[k] = tuple(b[k])
    return RunConfig(**b)


def _variants(q):
    return [(n, ov) for n, ov in VARIANTS if not q.get("variants") or n in q["variants"]]


def _smoke_wall(q, base):
    w = q.get("smoke_wall") or base.walls[0]
    return w if w in base.walls else base.walls[0]


def _coco(q):
    coco = q.get("coco_json") or None
    return coco if coco and os.path.exists(coco) else None


def _smoke_base(q, base, prepare=False):
    """Config for the smoke test (copies a few training tiles when prepare=True)."""
    smoke_root = os.path.join(base.experiments_root, "p2_smoke")
    tiles = os.path.join(smoke_root, "_tiles")
    if prepare:
        n_tiles = int(q.get("smoke_tiles", 32))
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
    return dataclasses.replace(base, experiments_root=smoke_root, n_epochs=1,
                               walls=[_smoke_wall(q, base)],
                               ortho_dir=os.path.join(tiles, "ortho"),
                               normalmap_dir=os.path.join(tiles, "normalmap"),
                               mask_dir=os.path.join(tiles, "mask"))


# ---------------------------------------------------------------------------
# children: one job each
def _job_morph(q, base, a):
    from skimage.morphology import binary_closing as sk_closing
    from . import paths, p2_morph
    from . import gapmetrics as gm
    try:
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        log(f"python {sys.version.split()[0]} | torch {torch.__version__} | device {dev}")
    except Exception:
        pass
    wall = _smoke_wall(q, base)
    gt = gm._load_class_mask(paths.test_mask_path(base, wall)) > 0
    ok1, nd1 = p2_morph.self_check(gt, base.kernel_radius, sk_closing)
    ok2, nd2 = True, 0
    pr = paths.segmentation_raw_path(dataclasses.replace(base, experiment_name=PAPER_ENS,
                                                         run_number=1), wall)
    if os.path.exists(pr):
        ok2, nd2 = p2_morph.self_check(gm._load_class_mask(pr) > 0, base.kernel_radius, sk_closing)
    return {"identical": bool(ok1 and ok2), "n_diff": [nd1, nd2], "wall": wall}


def _job_reference(q, base, a):
    from .config import make_run_id
    from . import paths
    from .p2_eval import eval_run
    root = base.experiments_root
    sum_dir = os.path.join(root, SUMMARY)
    ref_dir = os.path.join(sum_dir, "reference")
    coco = _coco(q)
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
    return {"reference_check": ref_check}


def _job_smoke(q, base, a):
    from . import paths
    from .train import train
    from .segment import segment
    from .p2_eval import eval_run
    ov = dict(VARIANTS)[a.variant]
    c = dataclasses.replace(_smoke_base(q, base), experiment_name=a.variant, run_number=1, **ov)
    log(f"[smoke] {a.variant}")
    train(c, force=True)
    segment(c, force=True)
    eval_run(c, os.path.join(paths.experiment_dir(c), "p2_eval"), _coco(q), do_roi=True,
             force=True, log=log)
    return {}


def _job_smoke_ens(q, base, a):
    from . import paths
    from .train import train
    from .ensemble import run_ensemble
    from .p2_eval import eval_run
    from .p2_summary import build_summary
    sbase = _smoke_base(q, base)
    ov = dict(VARIANTS)[a.variant]
    c2 = dataclasses.replace(sbase, experiment_name=a.variant, run_number=2, **ov)
    train(c2, force=True)
    run_ensemble(sbase, source_experiment=a.variant, out_experiment=a.variant + ENS_SUFFIX,
                 channel_variants=(7,), n_runs=2, force=True)
    ce = dataclasses.replace(sbase, experiment_name=a.variant + ENS_SUFFIX, run_number=1)
    eval_run(ce, os.path.join(paths.experiment_dir(ce), "p2_eval"), _coco(q), do_roi=False,
             force=True, log=log)
    names = [n for n, _ in _variants(q)]
    build_summary(sbase.experiments_root, names, PAPER_RUNS, PAPER_ENS, 1,
                  os.path.join(base.experiments_root, SUMMARY, "reference"), ENS_SUFFIX,
                  note="VORABTEST (1 Epoche, wenige Kacheln) – Zahlen bedeutungslos.")
    return {}


def _job_preflight(q, base, a):
    """Per-tile extras on ALL training tiles, exactly as train.py builds them."""
    from .data import load_training_masks, encode_masks, border_weight_map, tile_roi
    out = {}
    for mode in ("gap_wins", "nearest"):
        t0 = time.time()
        masks_enc, _ = encode_masks(load_training_masks(base, mode))
        t1 = time.time()
        if a.extra == "wmap":
            w = np.stack([border_weight_map(m, base.wmap_w0, base.wmap_sigma) for m in masks_enc])
            info = {"max": round(float(w.max()), 3), "share_gt2": round(float((w > 2).mean()), 5)}
        else:
            r_t = int(round(base.cldice_roi_radius_px * base.img_size / base.window_size[0]))
            r = np.stack([tile_roi(m, r_t) for m in masks_enc])
            info = {"radius_cells": r_t, "roi_share": round(float(r.mean()), 4)}
        info.update(tiles=int(len(masks_enc)), load_s=round(t1 - t0, 1), extra_s=round(time.time() - t1, 1))
        log(f"[preflight] {a.extra} on {mode} targets: {info}")
        out[mode] = info
    return {"extra": a.extra, "per_mode": out}


def _item_cfg(base, name, run):
    return dataclasses.replace(base, experiment_name=name, run_number=run, **dict(VARIANTS)[name])


def _job_item(q, base, a):
    from . import paths
    from .train import train
    from .segment import segment
    from .p2_eval import eval_run
    c = _item_cfg(base, a.variant, a.run)
    trained = not os.path.exists(paths.checkpoint_path(c))
    t0 = time.time()
    train(c)
    segment(c)
    eval_run(c, os.path.join(paths.experiment_dir(c), "p2_eval"), _coco(q), do_roi=True, log=log)
    return {"trained": trained, "seconds": round(time.time() - t0, 1)}


def _job_ensemble(q, base, a):
    from . import paths
    from .ensemble import run_ensemble
    from .p2_eval import eval_run
    n = a.variant
    run_ensemble(base, source_experiment=n, out_experiment=n + ENS_SUFFIX,
                 channel_variants=(7,), n_runs=int(q.get("n_runs", 5)))
    ce = dataclasses.replace(base, experiment_name=n + ENS_SUFFIX, run_number=1)
    eval_run(ce, os.path.join(paths.experiment_dir(ce), "p2_eval"), _coco(q), do_roi=False, log=log)
    return {}


def _job_summary(q, base, a):
    from .p2_summary import build_summary
    root = base.experiments_root
    build_summary(root, [n for n, _ in _variants(q)], PAPER_RUNS, PAPER_ENS,
                  int(q.get("n_runs", 5)), os.path.join(root, SUMMARY, "reference"), ENS_SUFFIX)
    return {}


JOBS = {"morph": _job_morph, "reference": _job_reference, "smoke": _job_smoke,
        "smoke_ens": _job_smoke_ens, "preflight": _job_preflight, "item": _job_item,
        "ensemble": _job_ensemble, "summary": _job_summary}


def _child_heartbeat(path, token, label, stop_evt, every_s=60.0):
    while not stop_evt.is_set():
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"t": time.time(), "pid": os.getpid(), "token": token, "job": label}, f)
        except Exception:
            pass
        stop_evt.wait(every_s)


def _test_fault(where, sum_dir, hb_stop):
    """Test hook, inert unless P2_TEST_FAULT is set (only used by the cloud tests).
    P2_TEST_FAULT="crash:item:p2_focal:1;hang:item:p2_bgw3:2;slow:...;pyerr:...;crash_once:..."
    """
    spec = os.environ.get("P2_TEST_FAULT", "")
    for part in filter(None, spec.split(";")):
        kind, _, target = part.partition(":")
        if target != where:
            continue
        if kind == "crash_once":
            marker = os.path.join(sum_dir, "test_fault_" + re.sub(r"\W", "_", where) + ".done")
            if os.path.exists(marker):
                continue
            open(marker, "w").close()
            kind = "crash"
        if kind == "crash":
            faulthandler._sigsegv()
        elif kind == "hang":       # heartbeat stops, process stays
            hb_stop.set()
            time.sleep(10 ** 6)
        elif kind == "slow":       # heartbeat goes on, process never finishes
            time.sleep(10 ** 6)
        elif kind == "pyerr":
            raise RuntimeError(f"P2_TEST_FAULT pyerr at {where}")


def _child_main(a):
    q = _load_q(a.cfg)
    base = _base_config(q)
    sum_dir = os.path.join(base.experiments_root, SUMMARY)
    where = ":".join(str(x) for x in (a.job, a.variant or a.extra, a.run) if x not in (None, ""))
    hb_stop = threading.Event()
    every = min(60.0, max(1.0, float(q.get("hb_stale_s", HB_STALE_S)) / 4))
    threading.Thread(target=_child_heartbeat,
                     args=(os.path.join(sum_dir, "child_heartbeat.txt"), a.token, where, hb_stop, every),
                     daemon=True).start()
    if not a.crash_file:
        os.makedirs(os.path.join(sum_dir, "crashes"), exist_ok=True)
        a.crash_file = os.path.join(sum_dir, "crashes", f"{datetime.now():%Y%m%d-%H%M%S}_manual.txt")
    crash_f = open(a.crash_file, "w", encoding="utf-8")
    faulthandler.enable(file=crash_f, all_threads=True)
    if a.limit_h:
        # shortly before the queue would kill a hanging child, write where it is
        faulthandler.dump_traceback_later(max(60.0, a.limit_h * 3600 - 600), repeat=False,
                                          file=crash_f, exit=False)
    res_path = os.path.join(sum_dir, "child_result.json")
    rc = 1
    try:
        if a.fast_closing:
            from . import p2_morph
            p2_morph.patch_evaluators()
        _test_fault(where, sum_dir, hb_stop)
        out = JOBS[a.job](q, base, a) or {}
        result = {"token": a.token, "ok": True, "result": out}
        rc = 0
    except Exception as e:
        _err(sum_dir, f"child {where}", e)
        result = {"token": a.token, "ok": False, "error": repr(e)[:500]}
    for attempt in range(10):  # Windows refuses os.replace while the queue reads the file
        try:
            with open(res_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(result, f, default=str)
            os.replace(res_path + ".tmp", res_path)
            break
        except OSError as e:
            if attempt == 9:
                log(f"(note) child result not written: {e!r}")
            time.sleep(0.5 + attempt)
    faulthandler.cancel_dump_traceback_later()
    faulthandler.disable()
    crash_f.close()
    try:
        if os.path.getsize(a.crash_file) == 0:
            os.remove(a.crash_file)
    except OSError:
        pass
    hb_stop.set()
    return rc


# ---------------------------------------------------------------------------
# queue side: start a child and watch it
class _Child:
    """The child as Popen sees it (on Windows possibly the Python launcher) and
    the real interpreter, whose pid the child reports in its heartbeat."""

    def __init__(self, proc):
        self.proc = proc
        self.real_pid = None
        self._h = None
        self._k32 = None

    def attach(self, pid):
        self.real_pid = pid
        if os.name == "nt" and pid != self.proc.pid:
            try:
                import ctypes
                from ctypes import wintypes
                k = ctypes.WinDLL("kernel32", use_last_error=True)
                k.OpenProcess.restype = wintypes.HANDLE
                k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
                k.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
                k.CloseHandle.argtypes = (wintypes.HANDLE,)
                h = k.OpenProcess(0x00100000 | 0x1000, False, int(pid))  # SYNCHRONIZE | QUERY_LIMITED
                if h:
                    self._h, self._k32 = h, k
            except Exception:
                self._h = None

    def poll(self):
        """Exit code of the real interpreter if it has ended, else None."""
        if self._h:
            try:
                import ctypes
                from ctypes import wintypes
                code = wintypes.DWORD(0)
                if self._k32.GetExitCodeProcess(self._h, ctypes.byref(code)):
                    return None if code.value == 259 else int(code.value)  # 259 = STILL_ACTIVE
            except Exception:
                pass
        return self.proc.poll()

    def kill(self):
        for pid in {self.real_pid, self.proc.pid} - {None}:
            try:
                if pid == self.proc.pid:
                    self.proc.kill()
                else:
                    os.kill(pid, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
            except Exception:
                pass

    def close(self):
        try:
            self.proc.wait(timeout=60)  # a launcher ends with its child
        except subprocess.TimeoutExpired:
            log(f"(note) launcher pid {self.proc.pid} still there after its child ended -> closing it")
            try:
                self.proc.kill()
                self.proc.wait(timeout=30)
            except Exception:
                pass
        if self._h:
            try:
                self._k32.CloseHandle(self._h)
            except Exception:
                pass
            self._h = None


def _crash_excerpt(txt, n=8):
    """First line of a faulthandler report plus the frames of the thread that
    crashed or hung ("Current thread"; the main thread for a timeout dump)."""
    lines = [ln.rstrip() for ln in txt.splitlines()]
    head = [ln.strip() for ln in lines if ln.strip() and not ln.startswith(" ")
            and "thread" not in ln.lower()][:1]
    blocks, cur = [], None
    for ln in lines:
        if ln.lower().startswith(("current thread", "thread ")):
            cur = [ln.strip()]
            blocks.append(cur)
        elif cur is not None and ln.strip().startswith("File "):
            cur.append(ln.strip())
    pick = next((b for b in blocks if b[0].lower().startswith("current thread")), None)
    if pick is None and blocks:  # timeout dump: the main thread is listed last
        pick = blocks[-1]
    return head + (pick[1:n + 1] if pick else [])


def _spawn(ctx, job, label, variant=None, run=None, extra=None):
    """Run one job in its own process under the watchdog. Returns (ok, info)."""
    sum_dir = ctx["sum_dir"]
    limit_h = float(ctx["limits_h"].get(job, 6))
    hb_path = os.path.join(sum_dir, "child_heartbeat.txt")
    res_path = os.path.join(sum_dir, "child_result.json")
    for p in (hb_path, res_path):
        _rm(p)
    crash_dir = os.path.join(sum_dir, "crashes")
    os.makedirs(crash_dir, exist_ok=True)
    tag = re.sub(r"[^\w.-]+", "_", label).strip("_")
    crash_file = os.path.join(crash_dir, f"{datetime.now():%Y%m%d-%H%M%S}_{tag}.txt")
    token = uuid.uuid4().hex
    cmd = [sys.executable, "-m", "amg_pipeline.p2_queue", "--job", job, "--token", token,
           "--crash-file", crash_file, "--limit-h", str(limit_h)]
    if variant:
        cmd += ["--variant", variant]
    if run:
        cmd += ["--run", str(run)]
    if extra:
        cmd += ["--extra", extra]
    if ctx.get("fast_closing"):
        cmd += ["--fast-closing"]
    cmd.append(ctx["cfg_path"])
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW: no console pops up
    sys.stdout.flush()
    t0 = time.time()
    try:
        with open(ctx["log_path"], "a", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, cwd=_REPO, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                    creationflags=flags)
    except Exception as e:
        _err(sum_dir, f"starting {label}", e)
        why = f"could not start the child process: {e!r}"
        log(f"!!! {label} FAILED: {why}")
        return False, {"exit": None, "rc": None, "reason": why, "seconds": 0.0, "result": {},
                       "error": None, "why": why}
    child = _Child(proc)
    reason, res_seen, rc = None, None, None
    while True:
        if child.real_pid is None:
            hb = _read_json(hb_path)
            if hb and hb.get("token") == token:
                child.attach(int(hb["pid"]))
        rc = child.poll()
        if rc is not None:
            break
        res = _read_json(res_path)
        if res and res.get("token") == token:  # the child is done and exiting
            res_seen = res_seen or time.time()
            if time.time() - res_seen > 60:
                log(f"(note) {label}: result written, but the process did not end within 60 s -> closing it")
                child.kill()
                time.sleep(2)
                rc = child.poll()
                break
        el = time.time() - t0
        hb = _read_json(hb_path)
        hb_t = hb["t"] if hb and hb.get("token") == token else None
        hb_age = time.time() - hb_t if hb_t else el
        if el > limit_h * 3600:
            reason = f"time limit {limit_h:g} h exceeded"
        elif hb_age > ctx["hb_stale_s"]:
            reason = f"no heartbeat for {hb_age / 60:.1f} min"
        if reason:
            log(f"[watchdog] {label}: {reason} -> stopping the child (pid {child.real_pid or proc.pid})")
            child.kill()
            time.sleep(2)
            rc = child.poll()
            break
        time.sleep(ctx.get("poll_s", 5))
    child.close()
    if rc is None:
        rc = proc.returncode
    el = time.time() - t0
    res = _read_json(res_path)
    res = res if res and res.get("token") == token else None
    ok = bool(res and res.get("ok")) and reason is None
    info = {"exit": describe_rc(rc), "rc": rc, "reason": reason, "seconds": round(el, 1),
            "result": (res or {}).get("result") or {}, "error": (res or {}).get("error")}
    crash_txt = ""
    try:
        if os.path.exists(crash_file) and os.path.getsize(crash_file) > 0:
            with open(crash_file, encoding="utf-8", errors="replace") as f:
                crash_txt = f.read()
            info["crash_file"] = crash_file
    except OSError:
        pass
    if ok:
        if rc not in (0, None):
            log(f"(note) {label}: finished, but exit code {describe_rc(rc)}")
        return True, info
    why = reason or (f"Python error {info['error']}" if info["error"] else f"crashed, exit {describe_rc(rc)}")
    info["why"] = why
    log(f"!!! {label} FAILED after {el / 60:.1f} min: {why}")
    for ln in _crash_excerpt(crash_txt):
        log(f"    {ln}")
    if crash_txt or not info["error"]:
        with open(os.path.join(sum_dir, "errors.log"), "a", encoding="utf-8") as f:
            f.write(f"\n===== {_now()} | {label} | {why} =====\n"
                    + (crash_txt if crash_txt else "(no faulthandler output)\n"))
    return False, info


# ---------------------------------------------------------------------------
def run(queue_cfg_path):
    from .config import make_run_id
    from . import paths

    q = _load_q(queue_cfg_path)
    base = _base_config(q)
    root = base.experiments_root
    sum_dir = os.path.join(root, SUMMARY)
    ref_dir = os.path.join(sum_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    st = Status(os.path.join(sum_dir, "status.json"))
    stop_file = os.path.join(sum_dir, "STOP")
    _rm(stop_file)
    queue_crash = open(os.path.join(sum_dir, "crash_queue.txt"), "a", encoding="utf-8")
    faulthandler.enable(file=queue_crash, all_threads=True)  # if the queue itself ever dies
    n_runs = int(q.get("n_runs", 5))
    if q.get("coco_json") and not _coco(q):
        log(f"(note) COCO JSON not found, polygon units skipped: {q.get('coco_json')}")
    variants = _variants(q)
    names = [n for n, _ in variants]
    limits = dict(LIMITS_H, **(q.get("limits_h") or {}))
    ctx = {"sum_dir": sum_dir, "cfg_path": os.path.abspath(queue_cfg_path),
           "log_path": os.path.join(sum_dir, "queue_log.txt"), "limits_h": limits,
           "hb_stale_s": float(q.get("hb_stale_s", HB_STALE_S)), "fast_closing": False,
           "poll_s": float(q.get("poll_s", 5))}

    hb_stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(os.path.join(sum_dir, "heartbeat.txt"), hb_stop),
                     daemon=True).start()
    _keep_awake(True)
    st.set(phase="0_checks", pid=os.getpid(), started=st.d.get("started") or _now(),
           last_start=_now(), variants=names, n_runs=n_runs, process_model="one child per step")

    def finish(code, **kw):
        if kw:
            st.set(**kw)
        hb_stop.set()
        _keep_awake(False)
        return code

    def summary():
        _spawn(ctx, "summary", "summary")

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
        return finish(2, phase="checks_failed", problems=problems)
    st.set(checks={"tiles": counts, "free_gb": round(free), "notes": notes})

    # ---- 1 morph check ----------------------------------------------------
    st.set(phase="1_morph_check", current="morph check", current_since=_now())
    ok, info = _spawn(ctx, "morph", "morph check")
    if ok and info["result"].get("identical"):
        ctx["fast_closing"] = True
        log(f"[morph] fast closing identical on {info['result'].get('wall')} (GT and paper prediction) -> using it")
        st.set(morph="fast (identical)")
    elif ok:
        log(f"[morph] fast closing differs ({info['result'].get('n_diff')} px) -> keeping skimage (slow)")
        st.set(morph=f"skimage (fast differed: {info['result'].get('n_diff')} px)")
    else:
        st.set(morph=f"skimage (check failed: {info.get('why')})")

    # ---- 2 reference ------------------------------------------------------
    if not q.get("skip_reference"):
        st.set(phase="2_reference", current="reference", current_since=_now())
        ok, info = _spawn(ctx, "reference", "reference")
        st.set(reference_check=info["result"].get("reference_check") if ok else f"failed: {info.get('why')}")
        summary()

    # ---- 3 smoke test -----------------------------------------------------
    excluded = dict(st.d.get("excluded_variants") or {}) if q.get("skip_smoke") else {}
    if not q.get("skip_smoke"):
        st.set(phase="3_smoke_test", current="smoke test", current_since=_now(),
               smoke=None, smoke_seconds=None, excluded_variants={}, preflight=None)
        t_smoke = time.time()
        try:
            _smoke_base(q, base, prepare=True)
        except Exception as e:
            _err(sum_dir, "smoke test: preparing tiles", e)
            return finish(1, phase="smoke_failed", smoke=f"FAILED preparing tiles: {e!r}")
        times = {}
        for name, _ov in variants:
            st.set(current=f"smoke {name}", current_since=_now())
            ok, info = _spawn(ctx, "smoke", f"smoke {name}", variant=name)
            if ok:
                times[name] = info["seconds"]
            else:
                excluded[name] = f"smoke test: {info['why']}"
        survivors = [n for n in names if n not in excluded]
        if not survivors:
            return finish(1, phase="smoke_failed", smoke="FAILED: every variant failed",
                          excluded_variants=excluded)
        smoke_notes = []
        ok, info = _spawn(ctx, "smoke_ens", "smoke ensemble", variant=survivors[0])
        if not ok:
            smoke_notes.append(f"ensemble/summary smoke failed: {info['why']}")
        preflight = {}
        needs = {"wmap": [n for n, ov in variants if ov.get("weight_map")],
                 "roi": [n for n, ov in variants if ov.get("region_loss") == "dice+cldice"]}
        for extra, users in needs.items():
            users = [n for n in users if n not in excluded]
            if not users:
                continue
            st.set(current=f"preflight {extra}", current_since=_now())
            ok, info = _spawn(ctx, "preflight", f"preflight {extra}", extra=extra)
            if ok:
                preflight[extra] = info["result"].get("per_mode")
            else:
                preflight[extra] = f"FAILED: {info['why']}"
                for n in users:
                    excluded[n] = f"preflight {extra} on all tiles: {info['why']}"
        survivors = [n for n in names if n not in excluded]
        if not survivors:
            return finish(1, phase="smoke_failed", smoke="FAILED: every variant failed",
                          excluded_variants=excluded, preflight=preflight)
        verdict = "passed" if not excluded and not smoke_notes else \
            f"passed with {len(excluded)} variant(s) left out" + (f"; {'; '.join(smoke_notes)}" if smoke_notes else "")
        log(f"[smoke] {verdict} in {time.time() - t_smoke:.0f} s | left out: {excluded or 'none'}")
        st.set(smoke=verdict, smoke_seconds=times, excluded_variants=excluded, preflight=preflight)

    active = [(n, ov) for n, ov in variants if n not in excluded]

    # ---- 4 main queue -----------------------------------------------------
    items = [(r, n) for r in range(1, n_runs + 1) for n, _ in active]
    all_items = [(r, n) for r in range(1, n_runs + 1) for n in names]

    def is_done(r, n):
        c = _item_cfg(base, n, r)
        rid = make_run_id(c)
        ev = os.path.join(paths.experiment_dir(c), "p2_eval")
        return (os.path.exists(paths.checkpoint_path(c))
                and all(os.path.exists(paths.segmentation_raw_path(c, w)) for w in c.walls)
                and os.path.exists(os.path.join(ev, f"{rid}_gapstone.csv"))
                and os.path.exists(os.path.join(ev, f"{rid}_separation.csv")))

    durations = [float(x) for x in (st.d.get("durations_s") or [])]
    if not durations and st.d.get("mean_hours_per_run"):  # status written by the first queue version
        durations = [float(st.d["mean_hours_per_run"]) * 3600] * max(1, int(st.d.get("items_done") or 1))
    failed = {}  # (run, name) -> reason

    def progress():
        done = sum(is_done(r, n) for r, n in all_items)
        rest = sum(not is_done(r, n) for r, n in items)
        eta = None
        if durations:
            eta = (datetime.now() + timedelta(seconds=rest * float(np.mean(durations)))).strftime("%Y-%m-%d %H:%M")
        return done, rest, eta

    def main_pass(todo, label):
        for r, n in todo:
            if os.path.exists(stop_file):
                log("[queue] STOP file found -> stopping gracefully")
                st.set(phase="stopped", current=None)
                return False
            if is_done(r, n):
                continue
            free = _free_gb(root)
            if free < MIN_FREE_GB:
                log(f"[queue] only {free:.0f} GB free -> stopping")
                st.set(phase="stopped_disk_full", current=None)
                return False
            st.set(phase=label, current=f"{n} run{r}", current_since=_now())
            log(f"===== {label}: {n} run{r} =====")
            ok, info = _spawn(ctx, "item", f"{n} run{r}", variant=n, run=r)
            if ok:
                failed.pop((r, n), None)
                if info["result"].get("trained"):
                    durations.append(info["seconds"])
            else:
                failed[(r, n)] = info["why"]
            done, rest, eta = progress()
            st.set(items_total=len(all_items), items_done=done,
                   items_failed=[f"{a} run{b}: {why}" for (b, a), why in failed.items()],
                   eta=eta, durations_s=durations,
                   mean_hours_per_run=round(float(np.mean(durations)) / 3600, 2) if durations else None)
            summary()
        return True

    if not main_pass(items, "4_main_queue"):
        return finish(0)
    if failed:
        log(f"[queue] retrying {len(failed)} failed run(s) once")
        if not main_pass(list(failed), "4b_retry"):
            return finish(0)

    # ---- 5 ensembles ------------------------------------------------------
    st.set(phase="5_ensembles", current=None)
    ens_failed = {}
    for n, ov in active:
        if os.path.exists(stop_file):
            return finish(0, phase="stopped", current=None)
        if not all(os.path.exists(paths.checkpoint_path(_item_cfg(base, n, r))) for r in range(1, n_runs + 1)):
            log(f"[ensemble] {n}: not all {n_runs} runs present -> skipped")
            continue
        st.set(current=f"{n}{ENS_SUFFIX}", current_since=_now())
        ok, info = _spawn(ctx, "ensemble", f"ensemble {n}", variant=n)
        if not ok:
            ens_failed[n] = info["why"]
        summary()

    # ---- 6 summary ----------------------------------------------------------
    summary()
    done, rest, _ = progress()
    log(f"[queue] finished: {done}/{len(all_items)} runs | failed: {failed or 'none'} | "
        f"left out: {excluded or 'none'} | ensembles failed: {ens_failed or 'none'}")
    return finish(0, phase="6_done", current=None, finished=_now(), items_done=done,
                  items_failed=[f"{a} run{b}: {why}" for (b, a), why in failed.items()],
                  ensembles_failed=ens_failed)


def _parse(argv):
    p = argparse.ArgumentParser(prog="python -m amg_pipeline.p2_queue")
    p.add_argument("cfg")
    p.add_argument("--job", choices=sorted(JOBS))
    p.add_argument("--token", default="")
    p.add_argument("--crash-file", default="")
    p.add_argument("--limit-h", type=float, default=0.0)
    p.add_argument("--variant", default=None)
    p.add_argument("--run", type=int, default=None)
    p.add_argument("--extra", default=None, choices=[None, "wmap", "roi"])
    p.add_argument("--fast-closing", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    _a = _parse(sys.argv[1:])
    sys.exit(_child_main(_a) if _a.job else run(_a.cfg))
