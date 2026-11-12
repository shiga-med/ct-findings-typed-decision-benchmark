import numpy as np
import pytest

from ctfindings import metrics as M
from ctfindings.thresholds import optimize


def test_score_counts_and_f1():
    y = np.array([[1, 0], [1, 1], [0, 0]])
    p = np.array([[1, 1], [0, 1], [0, 0]])
    overall, per = M.score(y, p)
    assert (overall["tp"], overall["fp"], overall["fn"], overall["tn"]) == (2, 1, 1, 2)
    assert overall["micro_f1"] == pytest.approx(4 / 6)
    assert overall["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert overall["exact_match"] == pytest.approx(1 / 3)
    assert overall["hamming_loss"] == pytest.approx(2 / 6)
    assert per[0]["specificity"] == 1.0
    assert M.mcc(0, 0, 5, 0) == 0.0  # zero denominator


def test_auroc_ties_and_missing_class():
    assert M.auroc([0, 1, 0, 1], [0.5, 0.5, 0.1, 0.9]) == pytest.approx(0.875)
    assert M.auroc([0, 0], [0.1, 0.2]) is None
    assert M.average_precision([1, 0, 1], [0.9, 0.8, 0.1]) == pytest.approx((1 + 2 / 3) / 2)


def test_calibration_and_missing_probabilities():
    brier, logloss, ece = M.calibration([1, 0], [1.0, 0.0])
    assert brier == 0 and ece == 0 and logloss < 1e-12
    y = np.array([[1, 0], [0, 1]])
    p = np.array([[0.9, np.nan], [0.2, 0.7]])
    result, _ = M.probability_metrics(y, p)
    assert result["probability_cells"] == 3 and result["coverage"] == 0.75


def test_bootstrap_is_reproducible_and_paired():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (30, 18))
    a = [M.case_counts(y, y)]
    b = [M.case_counts(y, 1 - y)]
    r1 = M.paired_bootstrap(a, b, 200, 7, 0.95, 11)
    assert r1 == M.paired_bootstrap(a, b, 200, 7, 0.95, 11)
    assert r1["difference"] == pytest.approx(1.0)
    assert r1["adjusted_low"] <= r1["ci95_low"] <= r1["ci95_high"] <= r1["adjusted_high"]


def test_thresholds_rules():
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    p = np.array([[0.3, 0.6], [0.4, 0.5], [0.2, 0.9], [0.1, 0.8]])
    thresholds, f1 = optimize(y, p)
    assert f1 == 1.0
    assert thresholds == [0.3, 0.8]
    # several perfect candidates (0.5, 0.9): the one closest to 0.5 wins
    assert optimize(np.array([[1], [0]]), np.array([[0.9], [0.1]]))[0] == [0.5]
    assert M.disagreement([np.array([[0, 1]]), np.array([[1, 1]])]) == [0.5]
    with pytest.raises(ValueError):
        optimize(y, p + 1)
