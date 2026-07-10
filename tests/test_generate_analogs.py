import pandas as pd
from scripts.generate_analogs import read_seeds, df_to_records


def test_read_seeds_skips_blanks(tmp_path):
    p = tmp_path / "s.smi"; p.write_text("CCO\n\n  \nc1ccccc1\n")
    assert read_seeds(str(p)) == ["CCO", "c1ccccc1"]


def test_df_to_records_dedups_by_analog_keeping_best_sim():
    df = pd.DataFrame({
        "smiles": ["CCO", "CCO", "CCN"],
        "target": ["seedA", "seedA", "seedB"],
        "score":  [0.4, 0.9, 0.5],
    })
    recs = df_to_records(df)
    by = {r["smiles"]: r for r in recs}
    assert set(by) == {"CCO", "CCN"}
    assert by["CCO"]["sim"] == 0.9  # best kept
    assert by["CCN"]["seed"] == "seedB"
