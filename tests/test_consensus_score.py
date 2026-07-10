import numpy as np
import pandas as pd
from scripts.consensus_score import build_frame, benchmark


def _frame():
    # Target GOOD:        both scorers rank knowns above randoms (AUROC 1.0 each).
    # Target SMINA_FAILS: smina partially INVERTED (~0.22) but Boltz perfect (~1.0).
    # Target BOLTZ_FAILS: mirror-swapped — Boltz partially inverted (~0.22), smina perfect.
    # Each single scorer collapses on ONE target, so worst["smina"] AND worst["boltz"] are
    # both low, while consensus stays decent on both failure targets. The headline metric
    # (consensus worst-case beats BOTH singles' worst-case) is therefore a real, non-vacuous
    # check — partial (not exact-mirror) failures keep rankmean clearly above the 0.5 tie.
    rows = []
    # GOOD target: 3 known (strong = very negative), 3 random (weak); scorers agree.
    for i, s in enumerate([-9, -8, -7]):
        rows.append(("GOOD", f"k{i}", True, s, s))
    for i, s in enumerate([-4, -3, -2]):
        rows.append(("GOOD", f"r{i}", False, s, s))
    # SMINA_FAILS: boltz perfect (knowns strong), smina partially inverted (AUROC ~0.22).
    #   (smina_score, boltz_score) per molecule; strength = -score.
    sf_known = [(-6.0, -9.0), (-5.0, -8.9), (-3.0, -8.8)]
    sf_rand = [(-7.0, -2.0), (-6.5, -2.1), (-4.0, -2.2)]
    for i, (sm, bo) in enumerate(sf_known):
        rows.append(("SMINA_FAILS", f"k{i}", True, sm, bo))
    for i, (sm, bo) in enumerate(sf_rand):
        rows.append(("SMINA_FAILS", f"r{i}", False, sm, bo))
    # BOLTZ_FAILS: roles swapped — smina perfect, boltz partially inverted (AUROC ~0.22).
    bf_known = [(-9.0, -6.0), (-8.9, -5.0), (-8.8, -3.0)]
    bf_rand = [(-2.0, -7.0), (-2.1, -6.5), (-2.2, -4.0)]
    for i, (sm, bo) in enumerate(bf_known):
        rows.append(("BOLTZ_FAILS", f"k{i}", True, sm, bo))
    for i, (sm, bo) in enumerate(bf_rand):
        rows.append(("BOLTZ_FAILS", f"r{i}", False, sm, bo))
    return pd.DataFrame(rows, columns=["target", "molecule", "is_known", "smina", "boltz"])


def test_benchmark_auroc_and_worstcase_rescue():
    out = benchmark(_frame(), min_known=3)
    # GOOD: everyone perfect
    assert out["per_target"]["GOOD"]["smina"] == 1.0
    assert out["per_target"]["GOOD"]["boltz"] == 1.0
    # SMINA_FAILS: smina inverted (<0.5), boltz ~1, consensus recovers above smina
    sf = out["per_target"]["SMINA_FAILS"]
    assert sf["smina"] < 0.5 and sf["boltz"] > 0.9 and sf["rankmean"] > sf["smina"]
    # BOLTZ_FAILS: mirror — boltz inverted (<0.5), smina ~1, consensus recovers above boltz
    bf = out["per_target"]["BOLTZ_FAILS"]
    assert bf["boltz"] < 0.5 and bf["smina"] > 0.9 and bf["rankmean"] > bf["boltz"]
    # THE headline metric: consensus worst-case beats BOTH single scorers' worst-case.
    assert out["worst"]["rankmean"] > max(out["worst"]["smina"], out["worst"]["boltz"])


def test_build_frame_inner_join_drops_unmatched():
    smina = pd.DataFrame({"target": ["T", "T"], "molecule": ["a", "b"],
                          "is_known": [True, False], "smina": [-8.0, -3.0]})
    boltz = pd.DataFrame({"target": ["T", "T"], "molecule": ["a", "c"], "boltz": [-7.0, -1.0]})
    f = build_frame(smina, boltz)
    assert list(f["molecule"]) == ["a"]   # only 'a' in both


def test_benchmark_skips_low_known_target():
    df = pd.DataFrame([("T", "k0", True, -9, -9), ("T", "r0", False, -2, -2),
                       ("T", "r1", False, -1, -1)],
                      columns=["target", "molecule", "is_known", "smina", "boltz"])
    out = benchmark(df, min_known=5)
    assert "T" in out["skipped"] and "T" not in out["per_target"]
