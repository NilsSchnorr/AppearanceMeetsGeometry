"""
p2_summary.py — collects the P2 results and compares them with the paper model.

Per variant: mean, standard deviation and number of finished runs (AllWalls
rows) for stone IoU, gap metrics, the paper's stone-level rates, the P2
separation measure and the stone area ratio; plus the 5-run ensemble. Each
value is compared with the published 7-channel model (per run: the five runs
of v2_yaw_correction-epsV1; ensemble: v2_baseline_ens) and flagged when the
difference exceeds the noise floor of 0.02 (project convention).

Outputs in <experiments_root>/p2_summary/:
  p2_runs.csv        one row per run (all variants and the paper model)
  p2_variants.csv    mean/sd/n per variant + deltas and flags
  p2_ensembles.csv   one row per ensemble + deltas and flags
  p2_summary.md      short report (German), regenerated after every run
"""

import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

NOISE = 0.02

# (column, label, direction): +1 higher is better, -1 lower is better
KEY = [
    ("IoU_mean_stones", "Stein-IoU", +1),
    ("gap_iou", "Fugen-IoU", +1),
    ("gap_precision", "Fugen-Precision", +1),
    ("gap_recall", "Fugen-Recall", +1),
    ("alone_rate_mask_c8", "allein", +1),
    ("merged_rate_mask_c8", "verschmolzen", -1),
    ("split_rate_mask_c8", "zerfallen", -1),
    ("stone_area_dev", "Flächenabw.", -1),
]
MORE = [
    ("IoU_Ashlar", +1), ("IoU_Polygonal", +1), ("IoU_Quarry", +1), ("macro_F1", +1),
    ("gap_f1", +1), ("gap_ratio_dev", -1), ("stone_area_ratio", 0), ("gap_ratio", 0),
    ("detection_rate_c90_s90", +1), ("merge_rate_c90_s90", -1),
    ("undetected_rate_mask_c8", -1),
    ("alone_rate_mask_c4", +1), ("merged_rate_mask_c4", -1), ("split_rate_mask_c4", -1),
    ("alone_rate_polygon_c8", +1), ("merged_rate_polygon_c8", -1), ("split_rate_polygon_c8", -1),
    ("alone_rate_polygon_c4", +1), ("merged_rate_polygon_c4", -1), ("split_rate_polygon_c4", -1),
]


def _runs_table(experiments_root, exp, eval_dir, manifest_exp=None, run_ids=None):
    """One row per run: ROI metrics (manifest) + gap/stone + separation (eval_dir)."""
    mexp = manifest_exp or exp
    mpath = os.path.join(experiments_root, mexp, "manifest.csv")
    parts = []
    if os.path.exists(mpath):
        mf = pd.read_csv(mpath)
        mf = mf[(mf["wall"] == "AllWalls") & (mf["channels"] == 7)]
        if run_ids is not None:
            mf = mf[mf["run_id"].isin(run_ids)]
        parts.append(mf[["run_id", "IoU_mean_stones", "IoU_Ashlar", "IoU_Polygonal",
                         "IoU_Quarry", "macro_F1"]].drop_duplicates("run_id").set_index("run_id"))
    gs_files = sorted(glob.glob(os.path.join(eval_dir, "*_gapstone.csv")))
    if gs_files:
        gs = pd.concat([pd.read_csv(f) for f in gs_files], ignore_index=True)
        gs = gs[gs["wall"] == "AllWalls"].set_index("run_id")
        cols = [c for c in ("gap_iou", "gap_precision", "gap_recall", "gap_f1", "gap_ratio",
                            "stone_area_ratio", "detection_rate_c90_s90", "merge_rate_c90_s90")
                if c in gs.columns]
        parts.append(gs[cols])
    sep_files = sorted(glob.glob(os.path.join(eval_dir, "*_separation.csv")))
    if sep_files:
        sp = pd.concat([pd.read_csv(f) for f in sep_files], ignore_index=True)
        sp = sp[(sp["wall"] == "AllWalls") & (sp["class"] == "alle")]
        wide = {}
        for _, r in sp.iterrows():
            d = wide.setdefault(r["run_id"], {})
            for k in ("alone", "merged", "split", "undetected"):
                d[f"{k}_rate_{r['unit']}_c{int(r['connectivity'])}"] = r[f"{k}_rate"]
        parts.append(pd.DataFrame.from_dict(wide, orient="index"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, axis=1, join="outer")
    df.index.name = "run_id"
    df = df.reset_index()
    if "stone_area_ratio" in df:
        df["stone_area_dev"] = (df["stone_area_ratio"] - 1).abs()
    if "gap_ratio" in df:
        df["gap_ratio_dev"] = (df["gap_ratio"] - 1).abs()
    return df


def _flag(delta, direction):
    if direction == 0 or pd.isna(delta) or abs(delta) <= NOISE:
        return ""
    return "▲" if delta * direction > 0 else "▼"


def _fmt(x, nd=3):
    return "–" if pd.isna(x) else f"{x:.{nd}f}".replace(".", ",")


def _fmt_d(d):
    return "" if pd.isna(d) else (f"{d:+.3f}".replace(".", ","))


def build_summary(experiments_root, variants, paper_runs, paper_ens, n_runs,
                  reference_dir, ensemble_suffix="_ens", out_name="p2_summary", note=""):
    out_dir = os.path.join(experiments_root, out_name)
    os.makedirs(out_dir, exist_ok=True)
    cols_all = [c for c, _, _ in KEY] + [c for c, _ in MORE]
    direction = {c: d for c, _, d in KEY}
    direction.update({c: d for c, d in MORE})

    # ---- per run ---------------------------------------------------------
    paper = _runs_table(experiments_root, paper_runs, os.path.join(reference_dir, paper_runs),
                        run_ids=[f"7ch_run{i}" for i in range(1, 6)])
    paper.insert(0, "variant", "PAPER")
    tables = [paper]
    for v in variants:
        t = _runs_table(experiments_root, v, os.path.join(experiments_root, v, "p2_eval"))
        if len(t):
            t.insert(0, "variant", v)
            tables.append(t)
    runs = pd.concat(tables, ignore_index=True)
    runs.to_csv(os.path.join(out_dir, "p2_runs.csv"), index=False)

    present = [c for c in cols_all if c in runs.columns]
    agg = runs.groupby("variant")[present].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_runs"] = runs.groupby("variant").size()
    ref = agg.loc["PAPER"] if "PAPER" in agg.index else None
    for c in present:
        if ref is not None:
            agg[f"{c}_delta"] = agg[f"{c}_mean"] - ref[f"{c}_mean"]
            agg[f"{c}_flag"] = [_flag(d, direction[c]) for d in agg[f"{c}_delta"]]
    order = [v for v in ["PAPER"] + list(variants) if v in agg.index]
    agg = agg.loc[order]
    agg.to_csv(os.path.join(out_dir, "p2_variants.csv"))

    # ---- ensembles -------------------------------------------------------
    etabs = []
    pe = _runs_table(experiments_root, paper_ens, os.path.join(reference_dir, paper_ens),
                     run_ids=["7ch_run1"])
    if len(pe):
        pe.insert(0, "variant", "PAPER")
        etabs.append(pe)
    for v in variants:
        ev = v + ensemble_suffix
        t = _runs_table(experiments_root, ev, os.path.join(experiments_root, ev, "p2_eval"),
                        run_ids=["7ch_run1"])
        if len(t):
            t.insert(0, "variant", v)
            etabs.append(t)
    ens = pd.concat(etabs, ignore_index=True) if etabs else pd.DataFrame()
    if len(ens):
        ens = ens.set_index("variant")
        if "PAPER" in ens.index:
            for c in [c for c in cols_all if c in ens.columns]:
                ens[f"{c}_delta"] = ens[c] - ens.loc["PAPER", c]
                ens[f"{c}_flag"] = [_flag(d, direction[c]) for d in ens[f"{c}_delta"]]
        ens.to_csv(os.path.join(out_dir, "p2_ensembles.csv"))

    _write_md(out_dir, agg, ens, variants, n_runs, note)
    return agg, ens


def _table(lines, df, variants, mean_suffix, n_col=None):
    head = "| Variante | " + ("Läufe | " if n_col else "") + " | ".join(l for _, l, _ in KEY) + " |"
    lines.append(head)
    lines.append("|" + "---|" * (len(KEY) + (2 if n_col else 1)))
    for v in ["PAPER"] + [v for v in variants if v in df.index]:
        if v not in df.index:
            continue
        r = df.loc[v]
        cells = [("Paper-Modell" if v == "PAPER" else v)]
        if n_col:
            cells.append(str(int(r[n_col])))
        for c, _, _ in KEY:
            mc = f"{c}{mean_suffix}"
            if mc not in r.index:
                cells.append("–")
                continue
            s = _fmt(r[mc])
            if v != "PAPER" and f"{c}_delta" in r.index and not pd.isna(r[f"{c}_delta"]):
                s += f" ({_fmt_d(r[f'{c}_delta'])}{r[f'{c}_flag']})"
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")


def _verdicts(df, variants, mean_suffix):
    def d(v, c):
        k = f"{c}_delta"
        return df.loc[v, k] if (v in df.index and k in df.columns) else np.nan
    out = []
    vs = [v for v in variants if v in df.index]
    both = [v for v in vs if d(v, "gap_iou") > NOISE and d(v, "IoU_mean_stones") > NOISE]
    joint = [v for v in vs if d(v, "gap_iou") > NOISE and d(v, "IoU_mean_stones") >= -NOISE and v not in both]
    sep = [v for v in vs if d(v, "alone_rate_mask_c8") > NOISE and d(v, "split_rate_mask_c8") <= NOISE]
    worse = [v for v in vs if d(v, "IoU_mean_stones") < -NOISE]
    more_split = [v for v in vs if d(v, "split_rate_mask_c8") > NOISE]
    out.append(f"- Stein und Fuge zugleich besser: {', '.join(both) or 'keine'}")
    out.append(f"- Fuge besser, Stein nicht schlechter: {', '.join(joint) or 'keine'}")
    out.append(f"- Mehr Steine allein, ohne mehr Zerfallen: {', '.join(sep) or 'keine'}")
    out.append(f"- Stein-IoU schlechter: {', '.join(worse) or 'keine'}")
    out.append(f"- Mehr zerfallene Steine: {', '.join(more_split) or 'keine'}")
    for c, label in (("gap_iou", "Fugen-IoU"), ("alone_rate_mask_c8", "Anteil Steine allein"),
                     ("IoU_mean_stones", "Stein-IoU")):
        mc = f"{c}{mean_suffix}"
        if mc in df.columns and vs:
            col = df.loc[vs, mc].astype(float).dropna()
            if len(col):
                best = col.idxmax()
                out.append(f"- Höchste {label}: {best} ({_fmt(col[best])})")
    return out


def _write_md(out_dir, agg, ens, variants, n_runs, note):
    L = [f"# P2 – Fugenläufe: Zusammenfassung",
         "",
         f"Automatisch erzeugt am {datetime.now():%d.%m.%Y %H:%M}. Vergleich mit dem veröffentlichten "
         f"7-Kanal-Modell. In Klammern die Differenz; ▲/▼ = besser/schlechter jenseits der "
         f"Rauschgrenze von 0,02. Alle Werte über alle vier Testmauern (AllWalls). "
         f"Vorläufig, nicht interpretiert.",
         ""]
    if note:
        L += [note, ""]
    L += ["## Je Lauf (Mittel über die fertigen Läufe)", ""]
    _table(L, agg, variants, "_mean", n_col="n_runs")
    unfinished = [v for v in variants if v in agg.index and agg.loc[v, "n_runs"] < n_runs]
    missing = [v for v in variants if v not in agg.index]
    if unfinished or missing:
        L += ["", f"Noch nicht alle {n_runs} Läufe: " + ", ".join(
            [f"{v} ({int(agg.loc[v, 'n_runs'])})" for v in unfinished] + [f"{v} (0)" for v in missing])]
    L += ["", "Kurz (je Lauf):", ""] + _verdicts(agg, variants, "_mean")
    if len(ens):
        L += ["", "## Ensembles (Mittel der Softmax über die Läufe)", ""]
        _table(L, ens, variants, "")
        L += ["", "Kurz (Ensembles):", ""] + _verdicts(ens, variants, "")
    L += ["",
          "## Lesehilfe",
          "",
          "- **Stein-IoU**: mittlere IoU der Steinklassen in der ROI (wie im Paper).",
          "- **Fugen-IoU, -Precision, -Recall**: Fugenpixel in der ROI (Closing r = 45), wie `gapmetrics.py`.",
          "- **allein / verschmolzen / zerfallen**: GT-Steine = Flächen der Maske ab 100 px, "
          "Komponenten der Vorhersage mit 8er-Nachbarschaft, ohne Klassen. Eine Komponente zählt für "
          "einen Stein, wenn sie mindestens 10 % von ihm bedeckt. 4er und Polygone stehen in `p2_variants.csv`.",
          "- **Flächenabw.**: Betrag von (vorhergesagte Steinfläche in der ROI / GT-Steinfläche − 1); 0 = gleich groß.",
          "- Paper-Modell je Lauf = `v2_yaw_correction-epsV1` 7ch, Läufe 1–5; Ensemble = `v2_baseline_ens`.",
          ""]
    with open(os.path.join(out_dir, "p2_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
