import numpy as np
import biotite.structure as struc
from synformer.dock.af_receptor import (
    MAX_CA_RMSD,
    MIN_ANCHORS_ABS,
    MIN_ANCHORS_FRAC,
    superpose_onto,
    pocket_mean_plddt,
)


_AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]


def _tiny_ca(coords, chain="A", res_names=None):
    n = len(coords)
    a = struc.AtomArray(n)
    a.coord = np.array(coords, dtype=float)
    a.chain_id = np.array([chain] * n)
    a.res_id = np.arange(1, n + 1)
    a.res_name = np.array(["GLY"] * n if res_names is None else list(res_names))
    a.atom_name = np.array(["CA"] * n)
    a.element = np.array(["C"] * n)
    return a


def _passes_gate(n_anchors, n_fixed_ca, n_mobile_ca, ca_rmsd):
    """Pure helper mirroring prepare_af_target's correspondence-quality gate, so the
    threshold logic can be unit-tested without hitting the network."""
    min_anchors_required = max(MIN_ANCHORS_ABS, MIN_ANCHORS_FRAC * min(n_fixed_ca, n_mobile_ca))
    return n_anchors >= min_anchors_required and ca_rmsd <= MAX_CA_RMSD


def test_superpose_recovers_translation():
    fixed = _tiny_ca([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    moved_in = fixed.copy(); moved_in.coord = moved_in.coord + np.array([10.0, 5.0, -3.0])
    out, rmsd, n_anchors = superpose_onto(fixed, moved_in, moved_in)
    assert rmsd < 1e-6
    assert n_anchors == 4
    assert np.allclose(out.coord, fixed.coord, atol=1e-5)


def test_superpose_full_length_mobile_vs_truncated_fixed():
    """Realistic case the toy 4-atom test misses: mobile is FULL-length (like an AF
    model) and fixed is a TRUNCATED subset (like a crystallized construct), so
    `fitted_ca` (full-length) and `fixed_ca[fixed_idx]` (anchor subset) differ in
    length and must be re-indexed by mobile_idx/fixed_idx to compare correctly."""
    # An irregular (non-collinear, non-uniformly-spaced) backbone-like path, so the
    # structural correspondence is unambiguous (an evenly-spaced straight line would
    # be translation-symmetric and admit multiple equally-good matches).
    n_mobile = 30
    rng = np.random.default_rng(0)
    steps = rng.normal(loc=[3.8, 0, 0], scale=[0.3, 1.2, 1.2], size=(n_mobile, 3))
    mobile_coords = np.cumsum(steps, axis=0)
    # Distinct per-residue names so superimpose_homologs has real SEQUENCE signal to
    # find the correct correspondence (all-GLY would be alignment-ambiguous and mis-pair
    # the truncated subset). A seeded 20-letter sequence makes the 20-mer subset unique.
    mobile_names = [_AA3[i] for i in rng.integers(0, len(_AA3), size=n_mobile)]
    mobile = _tiny_ca(mobile_coords, res_names=mobile_names)
    # fixed = a rigidly-transformed subset of mobile's residues (exact homology by
    # construction): residues 5..24 (20 residues), rotated+translated, SAME sequence.
    subset_idx = np.arange(5, 25)
    theta = np.pi / 6
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    translation = np.array([7.0, -2.0, 3.0])
    fixed_coords = mobile.coord[subset_idx] @ rot.T + translation
    fixed = _tiny_ca(fixed_coords, res_names=[mobile_names[i] for i in subset_idx])

    moved_full, rmsd, n_anchors = superpose_onto(fixed, mobile, mobile)

    assert n_anchors >= 15  # most/all of the 20-residue overlap should match as anchors
    assert rmsd < 1e-3
    # The matched mobile atoms, once moved, should land on the fixed frame (same
    # sequential correspondence since fixed was built as an ordered subset of mobile).
    assert np.allclose(moved_full.coord[subset_idx], fixed.coord, atol=1e-2)


def test_pocket_plddt_averages_bfactor_near_ligand(tmp_path):
    # AF protein: 2 residues, one at origin (b=90) one far (b=40); ligand at origin
    prot = _tiny_ca([[0, 0, 0], [50, 50, 50]])
    prot.b_factor = np.array([90.0, 40.0])
    lig = tmp_path / "lig.pdb"
    l = struc.AtomArray(1); l.coord = np.array([[0.0, 0.0, 0.0]]); l.chain_id = np.array(["L"])
    l.res_id = np.array([1]); l.res_name = np.array(["LIG"]); l.atom_name = np.array(["C1"])
    l.element = np.array(["C"]); l.hetero = np.array([True])
    import biotite.structure.io.pdb as pdb_io
    f = pdb_io.PDBFile(); pdb_io.set_structure(f, l); f.write(str(lig))
    # only the origin residue is within 8 A -> pLDDT ~90
    assert abs(pocket_mean_plddt(prot, str(lig), radius=8.0) - 90.0) < 1e-6


def test_pocket_plddt_is_per_residue_not_per_atom(tmp_path):
    """Multi-atom residues of differing sizes near the ligand: a 1-atom residue at
    pLDDT 50 and a 5-atom residue at pLDDT 90. Per-residue mean is (50+90)/2 = 70.0;
    the (buggy) atom-weighted mean would be (1*50 + 5*90) / 6 = 83.33."""
    # residue 1: a single atom at pLDDT 50; residue 2: five atoms at pLDDT 90.
    # All within 8 A of the ligand at the origin.
    n = 6
    prot = struc.AtomArray(n)
    prot.coord = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0], [5.0, 0, 0]])
    prot.chain_id = np.array(["A"] * n)
    prot.res_id = np.array([1, 2, 2, 2, 2, 2])
    prot.res_name = np.array(["GLY", "TRP", "TRP", "TRP", "TRP", "TRP"])
    prot.atom_name = np.array(["CA", "N", "CA", "C", "O", "CB"])
    prot.element = np.array(["C", "N", "C", "C", "O", "C"])
    prot.b_factor = np.array([50.0, 90.0, 90.0, 90.0, 90.0, 90.0])

    lig = tmp_path / "lig.pdb"
    l = struc.AtomArray(1); l.coord = np.array([[0.0, 0.0, 0.0]]); l.chain_id = np.array(["L"])
    l.res_id = np.array([1]); l.res_name = np.array(["LIG"]); l.atom_name = np.array(["C1"])
    l.element = np.array(["C"]); l.hetero = np.array([True])
    import biotite.structure.io.pdb as pdb_io
    f = pdb_io.PDBFile(); pdb_io.set_structure(f, l); f.write(str(lig))

    result = pocket_mean_plddt(prot, str(lig), radius=8.0)
    assert abs(result - 70.0) < 1e-6
    assert abs(result - 83.333) > 1.0  # not the atom-weighted mean


def test_gate_rejects_weak_correspondence():
    # 3 anchors out of 100/100-residue chains: far below both the absolute (20) and
    # fractional (0.4 * 100 = 40) thresholds -> reject even with a tiny rmsd.
    assert not _passes_gate(n_anchors=3, n_fixed_ca=100, n_mobile_ca=100, ca_rmsd=0.5)


def test_gate_rejects_large_rmsd_even_with_enough_anchors():
    assert not _passes_gate(n_anchors=50, n_fixed_ca=100, n_mobile_ca=100, ca_rmsd=6.0)


def test_gate_accepts_strong_correspondence():
    # e.g. RBP4-like: 131 anchors, 0.34 A rmsd, ~150-residue chains
    assert _passes_gate(n_anchors=131, n_fixed_ca=150, n_mobile_ca=201, ca_rmsd=0.34)
