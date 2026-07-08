import numpy as np
import biotite.structure as struc
from synformer.dock.af_receptor import superpose_onto, pocket_mean_plddt


def _tiny_ca(coords, chain="A"):
    n = len(coords)
    a = struc.AtomArray(n)
    a.coord = np.array(coords, dtype=float)
    a.chain_id = np.array([chain] * n)
    a.res_id = np.arange(1, n + 1)
    a.res_name = np.array(["GLY"] * n)
    a.atom_name = np.array(["CA"] * n)
    a.element = np.array(["C"] * n)
    return a


def test_superpose_recovers_translation():
    fixed = _tiny_ca([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    moved_in = fixed.copy(); moved_in.coord = moved_in.coord + np.array([10.0, 5.0, -3.0])
    out, rmsd = superpose_onto(fixed, moved_in, moved_in)
    assert rmsd < 1e-6
    assert np.allclose(out.coord, fixed.coord, atol=1e-5)


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
