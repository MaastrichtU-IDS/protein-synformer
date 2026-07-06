# tests/test_boltz_matrix.py
import json
import pandas as pd
from scripts.boltz_matrix import enumerate_cells, parse_results, cell_done


def test_enumerate_cells_is_full_grid():
    inputs = {
        "hits": [{"target_id": "A", "smiles": "CCC"}, {"target_id": "B", "smiles": "CCO"}],
        "proteins": [{"target_id": "A", "sequence": "AAA"}, {"target_id": "B", "sequence": "CCC"}],
    }
    cells = enumerate_cells(inputs)
    assert len(cells) == 4  # 2 hits x 2 proteins
    diag = [c for c in cells if c["hit_target"] == c["protein"]]
    assert len(diag) == 2
    a_into_b = [c for c in cells if c["hit_target"] == "A" and c["protein"] == "B"][0]
    assert a_into_b["smiles"] == "CCC" and a_into_b["sequence"] == "CCC"
    assert a_into_b["stem"] == "A_into_B"


def test_parse_results_reads_affinity_and_confidence(tmp_path):
    stem = "A_into_B"
    pdir = tmp_path / f"boltz_results_{stem}" / "predictions" / stem
    pdir.mkdir(parents=True)
    json.dump({"affinity_pred_value": -1.5, "affinity_probability_binary": 0.8},
              open(pdir / f"affinity_{stem}.json", "w"))
    json.dump({"ligand_iptm": 0.72}, open(pdir / f"confidence_{stem}_model_0.json", "w"))
    r = parse_results(str(tmp_path), stem)
    assert r["affinity_pred"] == -1.5 and r["binder_prob"] == 0.8 and r["ligand_iptm"] == 0.72


def test_parse_results_missing_is_nan(tmp_path):
    r = parse_results(str(tmp_path), "nope")
    assert r["affinity_pred"] != r["affinity_pred"]  # NaN


def test_cell_done_keys_on_pair(tmp_path):
    csv = tmp_path / "b.csv"
    pd.DataFrame([{"hit_target": "A", "protein": "B", "smiles": "CCC",
                   "affinity_pred": -1.5, "binder_prob": 0.8, "ligand_iptm": 0.7}]).to_csv(csv, index=False)
    assert cell_done(str(csv), "A", "B") is True
    assert cell_done(str(csv), "A", "A") is False
    assert cell_done(str(tmp_path / "missing.csv"), "A", "B") is False
