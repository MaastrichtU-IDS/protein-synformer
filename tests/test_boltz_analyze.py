import numpy as np
from scripts.boltz_analyze import normalized_summary, compare_matrices


def test_normalized_summary_flags_specificity():
    # own (diagonal) clearly best in its own column
    M = np.array([[-3.0, 0.0, 0.0],
                  [0.0, -3.0, 0.0],
                  [0.0, 0.0, -3.0]])
    s = normalized_summary(M)
    assert s["delta"] < 0        # own binds better than off-diagonal
    assert s["win_rate"] == 1.0


def test_compare_matrices_perfectly_correlated():
    A = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    B = A * 2.0  # monotonic
    c = compare_matrices(A, B)
    assert c["n"] == 4
    assert c["spearman"] > 0.99
    assert c["sign_agreement"] == 1.0  # same own<offdiag pattern


def test_compare_matrices_ignores_nan_cells():
    A = np.array([[-1.0, np.nan], [-3.0, -4.0]])
    B = np.array([[-2.0, -9.0], [-6.0, -8.0]])
    c = compare_matrices(A, B)
    assert c["n"] == 3  # the NaN cell in A is dropped from the correlation
