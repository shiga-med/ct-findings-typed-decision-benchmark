"""Metrics (paper section 2.6). Zero denominators give 0."""

import math
import statistics as st
from itertools import pairwise

import numpy as np

EPS = 1e-15


def ratio(a, b):
    return a / b if b else 0.0


def f1(tp, fp, fn):
    return ratio(2 * tp, 2 * tp + fp + fn)


def mcc(tp, fp, tn, fn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


def counts_metrics(tp, fp, tn, fn):
    return {"f1": f1(tp, fp, fn), "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "specificity": ratio(tn, tn + fp), "npv": ratio(tn, tn + fn),
            "mcc": mcc(tp, fp, tn, fn)}


def score(truth, predicted):
    """truth, predicted: case x finding 0/1 matrices. Returns overall and per-finding metrics."""
    y, p = np.asarray(truth), np.asarray(predicted)
    per = []
    for j in range(y.shape[1]):
        tp = int(((y[:, j] == 1) & (p[:, j] == 1)).sum())
        fp = int(((y[:, j] == 0) & (p[:, j] == 1)).sum())
        tn = int(((y[:, j] == 0) & (p[:, j] == 0)).sum())
        fn = int(((y[:, j] == 1) & (p[:, j] == 0)).sum())
        per.append({"positive": tp + fn, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    **counts_metrics(tp, fp, tn, fn)})
    tp, fp, tn, fn = (sum(v[k] for v in per) for k in ("tp", "fp", "tn", "fn"))
    overall = counts_metrics(tp, fp, tn, fn)
    overall = {"cases": len(y), "cells": y.size, "positive": tp + fn, "tp": tp, "fp": fp,
               "tn": tn, "fn": fn, "micro_f1": overall.pop("f1"),
               "macro_f1": st.fmean(v["f1"] for v in per), **overall,
               "exact_match": ratio(int((y == p).all(axis=1).sum()), len(y)),
               "hamming_loss": ratio(fp + fn, y.size)}
    return overall, per


def auroc(y, p):
    """Rank-based AUROC with average ranks for ties; None without both classes."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if not pos or not neg:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    sorted_p, i = p[order], 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if not int((y == 1).sum()):
        return None
    y = y[np.argsort(-p, kind="mergesort")]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def calibration(y, p, bins=10):
    """Brier score, log loss and ECE (equal-width bins, first bin closed at 0)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    brier = float(((p - y) ** 2).mean())
    c = np.clip(p, EPS, 1 - EPS)
    logloss = float(-(y * np.log(c) + (1 - y) * np.log(1 - c)).mean())
    ece = 0.0
    for lo, hi in pairwise(np.linspace(0, 1, bins + 1)):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return brier, logloss, float(ece)


def probability_metrics(truth, probabilities):
    """Discrimination and calibration over cells with a probability (NaN = missing)."""
    y, p = np.asarray(truth), np.asarray(probabilities, dtype=float)
    per_auc, per_ap, fy, fp = [], [], [], []
    by_label = []
    for j in range(y.shape[1]):
        ok = ~np.isnan(p[:, j])
        a, ap = auroc(y[ok, j], p[ok, j]), average_precision(y[ok, j], p[ok, j])
        by_label.append((a, ap))
        per_auc += [a] if a is not None else []
        per_ap += [ap] if ap is not None else []
        fy += y[ok, j].tolist()
        fp += p[ok, j].tolist()
    brier, logloss, ece = calibration(fy, fp)
    return {"probability_cells": len(fy), "coverage": len(fy) / y.size,
            "auroc_macro": st.fmean(per_auc) if per_auc else None, "auroc_findings": len(per_auc),
            "map": st.fmean(per_ap) if per_ap else None, "map_findings": len(per_ap),
            "brier": brier, "log_loss": logloss, "ece": ece}, by_label


# ---------------------------------------------------------------- bootstrap
def case_counts(truth, predicted):
    """Per-case (TP, FP, FN) summed over findings."""
    y, p = np.asarray(truth), np.asarray(predicted)
    return np.stack([((y == 1) & (p == 1)).sum(1), ((y == 0) & (p == 1)).sum(1),
                     ((y == 1) & (p == 0)).sum(1)], axis=1).astype(np.int64)


def paired_bootstrap(runs_a, runs_b, repetitions, seed, confidence, family):
    """Case-level paired bootstrap of the micro-F1 difference A - B.

    runs_*: one per-case (TP, FP, FN) array per run or seed. Each resample computes the
    micro-F1 of every run and averages them, as the point estimate does. Both conditions
    use the same resampled cases. Returns the point estimate and percentile intervals
    (unadjusted and Bonferroni-adjusted for `family` comparisons).
    """
    n = len(runs_a[0])
    if any(len(r) != n for r in runs_a + runs_b):
        raise ValueError("Conditions differ in the number of cases")
    idx = np.random.default_rng(seed).integers(0, n, (repetitions, n))

    def mean_f1(runs, ids=None):
        values = []
        for r in runs:
            s = r[ids].sum(axis=-2) if ids is not None else r.sum(axis=0)
            tp, fp, fn = s[..., 0], s[..., 1], s[..., 2]
            d = 2 * tp + fp + fn
            values.append(np.divide(2 * tp, d, out=np.zeros(np.shape(d)), where=d != 0))
        return np.mean(values, axis=0)

    deltas = mean_f1(runs_a, idx) - mean_f1(runs_b, idx)
    a = (1 - confidence) / 2
    lo, hi = np.quantile(deltas, [a, 1 - a])
    blo, bhi = np.quantile(deltas, [a / family, 1 - a / family])
    return {"difference": float(mean_f1(runs_a) - mean_f1(runs_b)), "ci95_low": float(lo),
            "ci95_high": float(hi), "adjusted_low": float(blo), "adjusted_high": float(bhi)}


def disagreement(predicted_runs):
    """Share of cells whose decision differs, for every pair of runs."""
    rates = []
    for i in range(len(predicted_runs)):
        for j in range(i + 1, len(predicted_runs)):
            a, b = np.asarray(predicted_runs[i]), np.asarray(predicted_runs[j])
            rates.append(float((a != b).mean()))
    return rates
