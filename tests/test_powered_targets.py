from scripts.powered_targets import (
    _ligand_atom_counts_from_structure,
    is_single_druglike_ligand,
    known_ligand_counts,
)
import pandas as pd
import biotite.structure as struc


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


def _make_atom(coord, chain_id, res_id, res_name, atom_name, element, hetero):
    return struc.Atom(
        coord,
        chain_id=chain_id,
        res_id=res_id,
        res_name=res_name,
        atom_name=atom_name,
        element=element,
        hetero=hetero,
    )


def test_ligand_atom_counts_from_structure_uses_largest_copy():
    atoms = []
    # a small amino-acid residue (protein backbone) that must NOT show up as a ligand
    for i, elem in enumerate(["N", "C", "C", "O"]):
        atoms.append(_make_atom([float(i), 0.0, 0.0], "A", 1, "ALA", f"{elem}{i}", elem, False))

    # ligand "LIG", first copy: chain A, 8 atoms total incl. 2 hydrogens -> 6 heavy atoms
    for i in range(6):
        atoms.append(_make_atom([float(i), 1.0, 0.0], "A", 10, "LIG", f"C{i}", "C", True))
    for i in range(2):
        atoms.append(_make_atom([float(i), 2.0, 0.0], "A", 10, "LIG", f"H{i}", "H", True))

    # ligand "LIG", second (larger) copy: chain B, 37 heavy atoms, no hydrogens
    for i in range(37):
        atoms.append(_make_atom([float(i), 3.0, 0.0], "B", 20, "LIG", f"C{i}", "C", True))

    arr = struc.array(atoms)
    counts = _ligand_atom_counts_from_structure(arr)

    assert counts == {"LIG": 37}   # largest copy wins, hydrogens excluded
    assert "ALA" not in counts     # amino-acid residues are excluded entirely
