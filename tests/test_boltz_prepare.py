# tests/test_boltz_prepare.py
import json
import pandas as pd
from scripts.boltz_matrix_prepare import top_hits, one_letter_from_residues, build_matrix_inputs


def test_top_hits_picks_lowest_own_pocket_candidate(tmp_path):
    csv = tmp_path / "s.csv"
    pd.DataFrame([
        {"target": "A", "pocket": "A", "molecule": "CCO", "source": "candidate", "score": -5.0},
        {"target": "A", "pocket": "A", "molecule": "CCC", "source": "candidate", "score": -8.0},  # best
        {"target": "A", "pocket": "A", "molecule": "CN",  "source": "known",     "score": -9.0},  # not a candidate
        {"target": "A", "pocket": "B", "molecule": "CCC", "source": "candidate", "score": -1.0},  # wrong pocket
    ]).to_csv(csv, index=False)
    assert top_hits(str(csv), ["A"], k=1) == {"A": ["CCC"]}


def test_one_letter_from_residues_maps_and_orders():
    # (res_ids, res_names) as biotite.get_residues returns, deliberately out of order
    res_ids = [3, 1, 2]
    res_names = ["GLY", "ALA", "CYS"]
    assert one_letter_from_residues(res_ids, res_names) == "ACG"


def test_one_letter_skips_nonstandard():
    res_ids = [1, 2, 3]
    res_names = ["ALA", "UNK", "GLY"]  # UNK -> X or skip; we skip unknown 3-letter codes
    assert one_letter_from_residues(res_ids, res_names) == "AG"


def test_build_matrix_inputs_shape(tmp_path, monkeypatch):
    import scripts.boltz_matrix_prepare as m
    monkeypatch.setattr(m, "pdb_to_sequence", lambda pdb_id: {"1AAA": "AAA", "2BBB": "CCC"}[pdb_id])
    tj = tmp_path / "t.json"; json.dump(
        [{"target_id": "A", "pdb_id": "1AAA", "ligand_resname": "LIG"},
         {"target_id": "B", "pdb_id": "2BBB", "ligand_resname": "LIG"}], open(tj, "w"))
    csv = tmp_path / "s.csv"
    pd.DataFrame([
        {"target": "A", "pocket": "A", "molecule": "CCC", "source": "candidate", "score": -8.0},
        {"target": "B", "pocket": "B", "molecule": "CCO", "source": "candidate", "score": -7.0},
    ]).to_csv(csv, index=False)
    out = tmp_path / "inputs.json"
    d = build_matrix_inputs(str(tj), str(csv), str(out), k=1)
    assert [h["target_id"] for h in d["hits"]] == ["A", "B"]
    assert d["hits"][0]["smiles"] == "CCC"
    assert {p["target_id"]: p["sequence"] for p in d["proteins"]} == {"A": "AAA", "B": "CCC"}
    assert json.load(open(out)) == d
