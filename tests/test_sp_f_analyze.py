import pandas as pd

from scripts.sp_f_analyze import compare_arms


def _row(target, arm, rnd, n_docked, best, top10):
    return {"target": target, "arm": arm, "round": rnd, "n_docked": n_docked,
            "best": best, "top10_mean": top10}


def test_compare_arms_final_round_deltas_and_parity():
    df = pd.DataFrame([
        _row("T1", "treatment", 0, 60, -8.0, -7.0),
        _row("T1", "treatment", 1, 60, -9.0, -8.5),
        _row("T1", "control_a", 0, 60, -8.0, -7.0),
        _row("T1", "control_a", 1, 60, -8.2, -7.8),
        _row("T1", "control_b", 0, 60, -8.0, -7.0),
        _row("T1", "control_b", 1, 60, -8.1, -7.5),
    ])
    out = compare_arms(df)
    t1 = out["per_target"]["T1"]
    assert t1["treatment"]["final_top10_mean"] == -8.5
    assert t1["treatment"]["best_overall"] == -9.0        # min best across rounds
    assert abs(t1["delta_treatment_minus_control_a"] - (-8.5 - -7.8)) < 1e-9   # -0.7
    assert abs(t1["delta_control_a_minus_control_b"] - (-7.8 - -7.5)) < 1e-9   # -0.3
    assert abs(t1["delta_treatment_minus_control_b"] - (-8.5 - -7.5)) < 1e-9   # -1.0
    assert out["parity_ok"] is True


def test_compare_arms_flags_docked_parity_violation():
    df = pd.DataFrame([
        _row("T1", "treatment", 0, 60, -8.0, -7.0),
        _row("T1", "control_a", 0, 45, -8.0, -7.0),   # docked fewer -> parity broken
    ])
    out = compare_arms(df)
    assert out["parity_ok"] is False
    assert out["parity_violations"] and out["parity_violations"][0]["target"] == "T1"
