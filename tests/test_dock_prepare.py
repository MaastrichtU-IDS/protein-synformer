"""Unit tests for the pure-logic drug-like ligand filter in dock_prepare.

Only the network/MPS-free logic is tested (per Task 4 brief): the HETATM
ignore-set + heavy-atom threshold, and the per-residue-copy aggregation.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dock_prepare import (  # noqa: E402
    MIN_HEAVY_ATOMS,
    find_drug_like_ligands,
    is_drug_like_ligand,
)


class TestIsDrugLikeLigand:
    def test_keeps_drug_like(self):
        # A typical drug-sized ligand.
        assert is_drug_like_ligand("STI", 40) is True
        assert is_drug_like_ligand("BTN", 16) is True

    def test_drops_water_and_ions(self):
        assert is_drug_like_ligand("HOH", 1) is False
        assert is_drug_like_ligand("NA", 1) is False
        assert is_drug_like_ligand("ZN", 1) is False
        assert is_drug_like_ligand("CL", 1) is False

    def test_drops_crystallization_additives(self):
        for add in ("GOL", "EDO", "PEG", "SO4", "PO4", "DMS", "MPD", "EPE"):
            assert is_drug_like_ligand(add, 20) is False, add

    def test_drops_common_sugars(self):
        assert is_drug_like_ligand("NAG", 14) is False
        assert is_drug_like_ligand("MAN", 12) is False

    def test_case_insensitive_ignore(self):
        assert is_drug_like_ligand("hoh", 1) is False
        assert is_drug_like_ligand("gol", 20) is False

    def test_heavy_atom_threshold(self):
        # Below threshold: dropped even if not an additive.
        assert is_drug_like_ligand("XYZ", MIN_HEAVY_ATOMS - 1) is False
        # At threshold: kept.
        assert is_drug_like_ligand("XYZ", MIN_HEAVY_ATOMS) is True

    def test_none_resname(self):
        assert is_drug_like_ligand(None, 100) is False


class TestFindDrugLikeLigands:
    def test_mixed_hetatms(self):
        # Build parallel per-atom arrays: a drug ligand LIG (chain A, res 500,
        # 20 atoms), a water HOH, and a glycerol GOL (14 atoms).
        res_names, res_ids, chain_ids = [], [], []
        for _ in range(20):
            res_names.append("LIG")
            res_ids.append(500)
            chain_ids.append("A")
        res_names.append("HOH")
        res_ids.append(600)
        chain_ids.append("A")
        for _ in range(14):
            res_names.append("GOL")
            res_ids.append(700)
            chain_ids.append("A")

        found = find_drug_like_ligands(res_names, res_ids, chain_ids)
        assert found == [("LIG", 20)]

    def test_uses_largest_copy(self):
        # Same ligand in two chains: a full 15-atom copy and a partial 5-atom
        # copy. The full copy should qualify it.
        res_names, res_ids, chain_ids = [], [], []
        for _ in range(15):
            res_names.append("DRG")
            res_ids.append(1)
            chain_ids.append("A")
        for _ in range(5):
            res_names.append("DRG")
            res_ids.append(1)
            chain_ids.append("B")
        found = find_drug_like_ligands(res_names, res_ids, chain_ids)
        assert found == [("DRG", 15)]

    def test_empty(self):
        assert find_drug_like_ligands([], [], []) == []

    def test_sorted_by_size_desc(self):
        res_names, res_ids, chain_ids = [], [], []
        for _ in range(13):
            res_names.append("SML")
            res_ids.append(1)
            chain_ids.append("A")
        for _ in range(30):
            res_names.append("BIG")
            res_ids.append(2)
            chain_ids.append("A")
        found = find_drug_like_ligands(res_names, res_ids, chain_ids)
        assert found == [("BIG", 30), ("SML", 13)]
