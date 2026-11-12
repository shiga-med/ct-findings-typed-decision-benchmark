"""Per-finding thresholds on the development set (paper section 2.5).

Coordinate ascent on the joint micro-F1 over all 18 findings, starting from 0.5.
Candidates: every observed probability plus 0, 0.5 and 1. Ties: closest to 0.5, then higher.
A finding is positive when probability >= threshold.
"""

import numpy as np


def micro_f1(truth, predicted):
    tp = int(((truth == 1) & (predicted == 1)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def optimize(truth, probabilities):
    y, p = np.asarray(truth), np.asarray(probabilities, dtype=float)
    if y.ndim != 2 or y.shape != p.shape or not len(y):
        raise ValueError("Expected two non-empty case x finding matrices of the same shape")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid labels or probabilities")
    n = y.shape[1]
    thresholds = np.full(n, 0.5)
    while True:  # finite candidates and strict improvement: always terminates
        before = thresholds.copy()
        for j in range(n):
            pred, other = p >= thresholds, np.arange(n) != j
            tp = int(((y[:, other] == 1) & pred[:, other]).sum())
            fp = int(((y[:, other] == 0) & pred[:, other]).sum())
            fn = int(((y[:, other] == 1) & ~pred[:, other]).sum())

            def objective(t, j=j, tp=tp, fp=fp, fn=fn):
                pos = p[:, j] >= t
                a = tp + int(((y[:, j] == 1) & pos).sum())
                b = fp + int(((y[:, j] == 0) & pos).sum())
                c = fn + int(((y[:, j] == 1) & ~pos).sum())
                return (2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0), -abs(t - 0.5), t

            thresholds[j] = max(sorted({0.0, 0.5, 1.0, *p[:, j].tolist()}), key=objective)
        if np.array_equal(before, thresholds):
            return thresholds.tolist(), micro_f1(y, p >= thresholds)
