from scripts.davis_analyze import summarize_pairs


def test_summarize_pairs_counts_positive():
    per_pair = {("A", "B"): 0.3, ("A", "C"): 0.1, ("B", "C"): -0.2}
    s = summarize_pairs(per_pair)
    assert s["n_pairs"] == 3 and s["n_positive"] == 2
    assert abs(s["median_rho"] - 0.1) < 1e-9
