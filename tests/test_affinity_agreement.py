from scripts.affinity_agreement import spearman


def test_spearman_monotonic():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) > 0.99
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) < -0.99


def test_spearman_constant_is_zero():
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0
