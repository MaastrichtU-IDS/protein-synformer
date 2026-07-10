import pandas as pd

from scripts.sp_f_boltz import top_m_from_dock, compare_boltz


def test_top_m_from_dock_unions_and_keeps_strongest(tmp_path):
    c0 = tmp_path / "r0.csv"; c0.write_text("smiles,score\nA,-7.0\nB,-5.0\n")
    c1 = tmp_path / "r1.csv"; c1.write_text("smiles,score\nA,-9.0\nC,-8.0\n")  # A improves to -9
    top = top_m_from_dock([str(c0), str(c1)], 2)
    assert top == ["A", "C"]          # A(-9) strongest, then C(-8); B(-5) excluded


def test_compare_boltz_delta_sign():
    df = pd.DataFrame([
        {"target": "T", "class": "treatment", "affinity_pred": 0.20},
        {"target": "T", "class": "treatment", "affinity_pred": 0.30},
        {"target": "T", "class": "control_b", "affinity_pred": 0.60},
        {"target": "T", "class": "control_b", "affinity_pred": 0.80},
    ])
    out = compare_boltz(df)["T"]
    assert out["treatment"]["mean_aff"] == 0.25
    assert out["control_b"]["best_aff"] == 0.60
    # treatment lower (stronger) than control_b -> negative delta = corroborates
    assert out["delta_mean_treatment_minus_control_b"] < 0
