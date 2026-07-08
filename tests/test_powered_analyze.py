import numpy as np
from scripts.powered_analyze import bootstrap_ci, paired_diff_ci


def test_bootstrap_ci_brackets_mean():
    vals = list(np.linspace(-2, 0, 100))  # mean = -1
    lo, hi = bootstrap_ci(vals, np.mean, n_boot=2000, seed=1)
    assert lo < -1.0 < hi
    assert hi - lo < 0.6  # reasonably tight for n=100


def test_paired_diff_ci_detects_shift():
    a = {f"t{i}": -2.0 for i in range(20)}   # crystal delta
    b = {f"t{i}": 0.0 for i in range(20)}    # AF delta (weaker specificity)
    mean, lo, hi = paired_diff_ci(a, b, seed=1)   # a - b
    assert mean < 0 and hi < 0     # crystal significantly more specific than AF


def test_paired_diff_ci_null_includes_zero():
    a = {f"t{i}": (-1.0 if i % 2 else 1.0) for i in range(20)}
    b = {f"t{i}": (1.0 if i % 2 else -1.0) for i in range(20)}
    mean, lo, hi = paired_diff_ci(a, b, seed=1)
    assert lo < 0 < hi
