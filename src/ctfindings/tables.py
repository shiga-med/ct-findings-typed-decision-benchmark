"""Evaluation of prediction files and the paper tables (paper sections 2.5-2.7, Tables 2-14).

A population is one prediction file (condition x language x text x set x run/seed).
Decisions: "development" applies the development thresholds to probabilities (or uses the
model's 0/1 answer for generative conditions); "fixed_0.5" applies 0.5. A missing decision
(failed request, refusal, invalid output) counts as negative; a missing probability stays
missing for discrimination and calibration.
"""

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics as M
from .data import DataError, cohort, load_labels, load_yaml, read_jsonl
from .runner import parse_name
from .thresholds import optimize

ID = ("set", "condition", "language", "text", "run")
GROUP = ("set", "condition", "language", "text")
COUNTS = ("cases", "cells", "positive", "tp", "fp", "tn", "fn")
OVERALL = ("micro_f1", "macro_f1", "precision", "recall", "specificity", "npv", "mcc",
           "exact_match", "hamming_loss")
PROB = ("probability_cells", "coverage", "auroc_macro", "auroc_findings", "map", "map_findings",
        "brier", "log_loss", "ece")
PER = ("positive", "tp", "fp", "tn", "fn", "f1", "precision", "recall", "specificity", "npv", "mcc")


# ---------------------------------------------------------------- populations
def base(condition):
    return condition.split("-frac")[0]


def fraction(condition):
    return condition.split("-frac")[1] if "-frac" in condition else ""


def thresholds_for(work, meta, frozen):
    folder = Path(work) / "thresholds"
    name = f"{meta['condition']}.{meta['language']}"
    for path in (folder / f"{name}.{meta['run']}.json", folder / f"{name}.json"):
        if path.exists():
            return json.loads(path.read_text())["thresholds"]
    if meta["condition"] in frozen:
        return frozen[meta["condition"]][meta["language"]]
    raise DataError(f"No thresholds for {name}; run `ctfindings thresholds` first")


def load_populations(work, config, sets=("test150", "external")):
    labels, _ = load_labels(config)
    frozen = load_yaml(Path(config) / "frozen_thresholds.yaml")
    cohorts = {s: cohort(work, s) for s in sets}
    populations = []
    for path in sorted((Path(work) / "predictions").glob("*.jsonl")):
        meta = parse_name(path)
        if meta["set"] not in sets:
            continue
        rows = {r["case_id"]: r for r in read_jsonl(path)}
        cases = cohorts[meta["set"]]
        if set(rows) != {c["case_id"] for c in cases}:
            raise DataError(f"Incomplete prediction file: {path.name}")
        y = np.asarray([c["labels"] for c in cases])
        raw = [rows[c["case_id"]]["predictions"] for c in cases]
        probs = [rows[c["case_id"]].get("probabilities") for c in cases]
        pop = {**meta, "y": y, "missing": sum(v is None for r in raw for v in r),
               "recorded": np.asarray([[v or 0 for v in r] for r in raw]), "p": None,
               "usage": [rows[c["case_id"]] for c in cases]}
        if any(p is not None for p in probs):
            pop["p"] = np.asarray([[np.nan if v is None else v for v in (p or [None] * y.shape[1])]
                                   for p in probs], dtype=float)
            pop["t"] = np.asarray(thresholds_for(work, meta, frozen))
        populations.append(pop)
    return labels, populations


def decisions(pop):
    """{threshold name: 0/1 matrix}."""
    if pop["p"] is None:
        return {"development": pop["recorded"]}
    ok = ~np.isnan(pop["p"])
    return {"development": np.where(ok, pop["p"] >= pop["t"], pop["recorded"]).astype(int),
            "fixed_0.5": np.where(ok, pop["p"] >= 0.5, 0).astype(int)}


def evaluate(work, config):
    """work/results/metrics.csv (population x threshold) and findings.csv."""
    labels, pops = load_populations(work, config)
    overall, per_rows = [], []
    for pop in pops:
        ident = {k: pop[k] for k in ID}
        prob, by_label = M.probability_metrics(pop["y"], pop["p"]) if pop["p"] is not None \
            else ({}, [(None, None)] * len(labels))
        for name, decided in decisions(pop).items():
            agg, per = M.score(pop["y"], decided)
            overall.append({**ident, "threshold": name, "missing_decisions": pop["missing"],
                            **{k: agg[k] for k in COUNTS + OVERALL},
                            **{k: prob.get(k, "") for k in PROB}})
            if name == "development":
                per_rows += [{**ident, "finding": x, **{k: v[k] for k in PER},
                              "auroc": "" if a is None else a, "ap": "" if ap is None else ap}
                             for x, v, (a, ap) in zip(labels, per, by_label, strict=True)]
    out = Path(work) / "results"
    write_csv(out / "metrics.csv", overall)
    write_csv(out / "findings.csv", per_rows)
    return {"populations": len(pops), "rows": len(overall)}


# ---------------------------------------------------------------- tables
def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    def value(v):
        try:
            return float(v) if v != "" else ""
        except ValueError:
            return v
    with open(path, encoding="utf-8") as fh:
        return [{k: (v if k in ID + ("threshold", "finding") else value(v)) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def run_order(run):
    return int(run.removeprefix("run").removeprefix("seed"))


def aggregate(rows, extra=()):
    """Mean over runs/seeds, with the micro-F1 range."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in GROUP + extra)].append(r)
    out = []
    for key, items in groups.items():
        items.sort(key=lambda r: run_order(r["run"]))
        row = {**dict(zip(GROUP + extra, key)), "runs": len(items)}
        for k, v in items[0].items():
            if k not in ID + extra and isinstance(v, float):
                values = [i[k] for i in items if i[k] != ""]
                row[k] = st.fmean(values) if values else ""
        if "micro_f1" in row:
            f = [i["micro_f1"] for i in items]
            row.update({"micro_f1_min": min(f), "micro_f1_max": max(f),
                        "micro_f1_range": max(f) - min(f)})
        out.append(row)
    return out


def tables(work, config):
    _, pops = load_populations(work, config)
    res = Path(work) / "results"
    metric_rows, finding_rows = read_csv(res / "metrics.csv"), read_csv(res / "findings.csv")
    agg = aggregate(metric_rows, ("threshold",))
    main = [r for r in agg if not fraction(r["condition"])]
    dev = {tuple(r[k] for k in GROUP): r for r in main if r["threshold"] == "development"}
    t = {}
    t["table02_03_internal.csv"] = [r for r in main if r["set"] == "test150"]
    t["table04_external.csv"] = [r for r in main if r["set"] == "external"]
    t["table06_internal_to_external.csv"] = [
        {"condition": e["condition"], "internal_text": text,
         **{f"{side}_{m}": v for m in ("micro_f1", "macro_f1", "exact_match", "precision",
                                       "recall", "specificity", "mcc")
            for side, v in (("internal", i[m]), ("external", e[m]), ("difference", e[m] - i[m]))}}
        for e in (r for r in dev.values() if r["set"] == "external")
        for text in ("ja_Checked", "ja_MT") if ("test150", e["condition"], "ja", text) in dev
        for i in [dev[("test150", e["condition"], "ja", text)]]]
    findings = aggregate([r for r in finding_rows if not fraction(r["condition"])], ("finding",))
    t["table07_findings_internal.csv"] = [r for r in findings if r["set"] == "test150"]
    t["table07_findings_external.csv"] = [r for r in findings if r["set"] == "external"]
    t["table08_fixed05_difference.csv"] = [
        {**{k: r[k] for k in GROUP},
         **{f"{name}_{m}": v for m in ("micro_f1", "macro_f1")
            for name, v in (("development", dev[k][m]), ("fixed05", r[m]),
                            ("difference", r[m] - dev[k][m]))}}
        for r in main if r["threshold"] == "fixed_0.5" for k in [tuple(r[c] for c in GROUP)]]
    t["table09_calibration.csv"] = [
        {**{k: r[k] for k in GROUP}, "runs": r["runs"], **{k: r[k] for k in PROB}}
        for r in main if r["threshold"] == "development" and r.get("probability_cells", "") != ""]
    t["table10_efficiency.csv"] = efficiency(pops)
    t["table11_variability.csv"] = [
        {**{k: r[k] for k in GROUP}, "axis": "seed" if r["condition"][0] == "E" else "run",
         "runs": r["runs"], **{k: r[k] for k in ("micro_f1", "micro_f1_min", "micro_f1_max",
                                                 "micro_f1_range")}}
        for r in main if r["threshold"] == "development" and r["runs"] > 1]
    t["table11_run_agreement.csv"] = run_agreement(pops)
    t["table12_comparisons.csv"] = comparisons(pops, agg, config)
    t["table13_training_fraction.csv"] = [
        {**r, "condition": base(r["condition"]), "fraction": fraction(r["condition"]) or "1"}
        for r in agg if r["condition"].startswith("E")]
    t["table14_threshold_refit.csv"], t["table14_threshold_refit_runs.csv"] = refit(pops)
    for name, rows in t.items():
        write_csv(res / name, rows)
    return {"tables": sorted(t)}


def efficiency(pops):
    """Tokens and request seconds per report, first run (or seed) of each population."""
    first = {}
    for pop in sorted(pops, key=lambda p: run_order(p["run"])):
        first.setdefault(tuple(pop[k] for k in GROUP), pop)
    rows = []
    for key, pop in first.items():
        u, n = pop["usage"], len(pop["usage"])
        if not any(r.get("requests") for r in u):
            continue  # encoders: no request log
        total = {k: sum(r.get(k) or 0 for r in u) for k in ("requests", "input_tokens",
                                                            "output_tokens", "seconds")}
        rows.append({**dict(zip(GROUP, key)), "run": pop["run"], "requests": total["requests"],
                     "input_tokens_per_report": total["input_tokens"] / n,
                     "output_tokens_per_report": total["output_tokens"] / n,
                     "seconds_per_report": total["seconds"] / n})
    return rows


def run_agreement(pops):
    """Share of cells whose decision changes between two runs (all pairs)."""
    groups = defaultdict(list)
    for pop in pops:
        if not fraction(pop["condition"]):
            groups[tuple(pop[k] for k in GROUP)].append(pop)
    rows = []
    for key, ps in groups.items():
        if len(ps) < 2:
            continue
        ps.sort(key=lambda p: run_order(p["run"]))
        rates = M.disagreement([decisions(p)["development"] for p in ps])
        rows.append({**dict(zip(GROUP, key)), "axis": "seed" if key[1][0] == "E" else "run",
                     "runs": len(ps), "pairs": len(rates), "cells": ps[0]["y"].size,
                     "disagreement_mean": st.fmean(rates), "disagreement_min": min(rates),
                     "disagreement_max": max(rates)})
    return rows


def comparisons(pops, agg, config):
    """Table 12: paired case-level bootstrap of the micro-F1 difference."""
    settings = load_yaml(Path(config) / "analysis.yaml")
    b, pairs = settings["bootstrap"], settings["comparisons"]
    mean = {(r["condition"], r["text"]): r["micro_f1"] for r in agg
            if r["set"] == "test150" and r["threshold"] == "development"}
    rows = []
    for number, (text, a, c) in enumerate(pairs, 1):
        runs = {}
        for cond in (a, c):
            ps = sorted((p for p in pops if p["set"] == "test150" and p["condition"] == cond
                         and p["text"] == text), key=lambda p: run_order(p["run"]))
            runs[cond] = [M.case_counts(p["y"], decisions(p)["development"]) for p in ps]
        if not runs[a] or not runs[c]:
            continue  # predictions not available for this comparison
        r = M.paired_bootstrap(runs[a], runs[c], b["repetitions"], b["seed"], b["confidence"],
                               len(pairs))
        judge = [("A higher" if lo > 0 else "B higher" if hi < 0 else "no difference")
                 for lo, hi in ((r["ci95_low"], r["ci95_high"]),
                                (r["adjusted_low"], r["adjusted_high"]))]
        rows.append({"number": number, "text": text, "condition_a": a, "condition_b": c,
                     "runs_a": len(runs[a]), "runs_b": len(runs[c]),
                     "micro_f1_a": mean[a, text], "micro_f1_b": mean[c, text], **r,
                     "judgement_95": judge[0], "judgement_changes_with_bonferroni":
                     judge[0] != judge[1], "cases": len(runs[a][0]),
                     "repetitions": b["repetitions"], "seed": b["seed"]})
    return rows


def refit(pops):
    """Table 14 (post hoc, upper bound): thresholds re-estimated on the external set itself."""
    per_run = []
    for pop in pops:
        if pop["set"] != "external" or pop["p"] is None or fraction(pop["condition"]):
            continue
        if np.isnan(pop["p"]).any():
            raise DataError(f"Missing probabilities: {pop['condition']} {pop['run']}")
        t, _ = optimize(pop["y"], pop["p"])
        refitted, _ = M.score(pop["y"], (pop["p"] >= np.asarray(t)).astype(int))
        frozen, _ = M.score(pop["y"], decisions(pop)["development"])
        per_run.append({"condition": pop["condition"], "run": pop["run"],
                        "frozen_micro_f1": frozen["micro_f1"],
                        "refit_micro_f1": refitted["micro_f1"],
                        "frozen_macro_f1": frozen["macro_f1"],
                        "refit_macro_f1": refitted["macro_f1"],
                        "refit_thresholds": " ".join(f"{v:g}" for v in t)})
    groups = defaultdict(list)
    for r in per_run:
        groups[r["condition"]].append(r)
    rows = []
    for cond, rs in groups.items():
        m = {k: st.fmean(r[k] for r in rs) for k in ("frozen_micro_f1", "refit_micro_f1",
                                                     "frozen_macro_f1", "refit_macro_f1")}
        rows.append({"condition": cond, "runs": len(rs), **m,
                     "recovery_micro_f1": m["refit_micro_f1"] - m["frozen_micro_f1"],
                     "refit_micro_f1_min": min(r["refit_micro_f1"] for r in rs),
                     "refit_micro_f1_max": max(r["refit_micro_f1"] for r in rs)})
    return rows, per_run


def thresholds(work, name, language):
    """Development thresholds from dev450 predictions (one file per run or seed)."""
    from .data import write_json
    dev450 = cohort(work, "dev450")
    y, ids = np.asarray([c["labels"] for c in dev450]), [c["case_id"] for c in dev450]
    written = []
    for path in sorted((Path(work) / "predictions").glob(f"{name}.{language}.*.dev450.*.jsonl")):
        rows = {r["case_id"]: r for r in read_jsonl(path)}
        if set(rows) != set(ids) or any(rows[i].get("probabilities") is None for i in ids):
            raise DataError(f"Development probabilities incomplete: {path.name}")
        p = np.asarray([rows[i]["probabilities"] for i in ids], dtype=float)
        t, f = optimize(y, p)
        run = parse_name(path)["run"]
        out = Path(work) / "thresholds" / (f"{name}.{language}.json" if run == "run1"
                                           else f"{name}.{language}.{run}.json")
        write_json(out, {"thresholds": t, "dev450_micro_f1": f, "source": path.name})
        written.append(str(out))
    if not written:
        raise DataError(f"No dev450 predictions for {name} {language}")
    return {"written": written}
