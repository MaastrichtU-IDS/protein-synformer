import numpy as np
from scripts.powered_analyze import bootstrap_ci, paired_diff_ci, _delta_win_from_matrix


def test_delta_win_from_matrix_own_pocket_preference():
    # Every source docks far better (more negative) into its OWN pocket than any source
    # docks into that pocket otherwise: diagonal = -10, everything else = -5. Column-wise
    # z-normalization should then show every own-pocket cell as the clear column minimum,
    # giving delta < 0 (own beats off-diagonal) and a perfect 1.0 win-rate for all sources.
    n = 5
    target_ids = [f"T{i}" for i in range(n)]
    M = np.full((n, n), -5.0)
    for i in range(n):
        M[i, i] = -10.0

    delta, win = _delta_win_from_matrix(M, target_ids)

    assert set(delta.keys()) == set(target_ids)
    assert all(d < 0 for d in delta.values())
    assert all(w == 1.0 for w in win.values())
    assert np.mean(list(win.values())) == 1.0


def test_delta_win_from_matrix_null_no_own_pocket_preference():
    # No association between source and pocket: i.i.d. noise, no systematically-better
    # diagonal. Column-normalized delta should average out near zero and the win-rate
    # should sit near chance (0.5), not near the 1.0 seen in the own-preference case.
    rng = np.random.default_rng(42)
    n = 20
    target_ids = [f"T{i}" for i in range(n)]
    M = rng.normal(loc=-6.0, scale=1.0, size=(n, n))

    delta, win = _delta_win_from_matrix(M, target_ids)

    assert len(delta) == n
    assert abs(np.mean(list(delta.values()))) < 0.6
    win_mean = np.mean(list(win.values()))
    assert 0.25 <= win_mean <= 0.75


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
