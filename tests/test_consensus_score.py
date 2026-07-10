import numpy as np
import pandas as pd
from scripts.consensus_score import build_frame, benchmark


def _frame():
    # Target GOOD: both scorers rank knowns above randoms (AUROC 1.0 each).
    # Target SMINA_FAILS: smina INVERTED (knowns look weak by smina) but Boltz correct;
    #   consensus should beat smina here (worst-case rescue).
    rows = []
    # GOOD target: 3 known (strong = very negative), 3 random (weak)
    for i, s in enumerate([-9, -8, -7]):
        rows.append(("GOOD", f"k{i}", True, s, s))       # smina & boltz agree strong
    for i, s in enumerate([-4, -3, -2]):
        rows.append(("GOOD", f"r{i}", False, s, s))
    # SMINA_FAILS target: knowns are Boltz-strong but smina-weak; randoms the opposite
    for i in range(3):
        rows.append(("SMINA_FAILS", f"k{i}", True, -2.0 - i * 0.1, -9.0 + i * 0.1))  # smina weak, boltz strong
    for i in range(3):
        rows.append(("SMINA_FAILS", f"r{i}", False, -9.0 + i * 0.1, -2.0 - i * 0.1))  # smina strong, boltz weak
    return pd.DataFrame(rows, columns=["target", "molecule", "is_known", "smina", "boltz"])


def test_benchmark_auroc_and_worstcase_rescue():
    out = benchmark(_frame(), min_known=3)
    # GOOD: everyone perfect
    assert out["per_target"]["GOOD"]["smina"] == 1.0
    assert out["per_target"]["GOOD"]["boltz"] == 1.0
    # SMINA_FAILS: smina AUROC ~0 (inverted), boltz ~1, consensus in between but > smina
    sf = out["per_target"]["SMINA_FAILS"]
    assert sf["smina"] < 0.5 and sf["boltz"] > 0.9
    assert sf["rankmean"] > sf["smina"]
    # worst-case (min across targets): consensus rescues smina's catastrophic target
    assert out["worst"]["rankmean"] > out["worst"]["smina"]


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
