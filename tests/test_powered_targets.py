from scripts.powered_targets import is_single_druglike_ligand, known_ligand_counts
import pandas as pd


def test_single_druglike_ligand_returns_sole_ligand():
    # one drug-like (STI, 30 heavy atoms), waters/ions ignored
    counts = {"STI": 30, "HOH": 1, "NA": 1, "SO4": 5}
    assert is_single_druglike_ligand(counts) == "STI"


def test_single_druglike_ligand_rejects_multiple():
    counts = {"STI": 30, "DEX": 28}   # two drug-like -> ambiguous pocket
    assert is_single_druglike_ligand(counts) is None


def test_single_druglike_ligand_rejects_none():
    counts = {"HOH": 3, "GOL": 6, "NA": 1}  # only additives/small
    assert is_single_druglike_ligand(counts) is None


def test_known_ligand_counts(tmp_path):
    csv = tmp_path / "t.csv"
    pd.DataFrame({"SMILES": ["CCO", "CCC", "CCO", "CN"],
                  "target_id": ["A_WT", "A_WT", "A_WT", "B_WT"]}).to_csv(csv, index=False)
    c = known_ligand_counts(str(csv))
    assert c["A"] == 2 and c["B"] == 1   # unique SMILES per accession
