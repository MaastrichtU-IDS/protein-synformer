"""Extract a binding pocket (residues near the co-crystal ligand) from a holo structure,
as per-residue CA/Cβ coords + residue-type indices — the input to the 3D pocket encoder."""
from __future__ import annotations

import numpy as np
import biotite.structure as struc
from scipy.spatial import cKDTree

_AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA3_TO_IDX = {a: i for i, a in enumerate(_AA3)}


def pocket_residues(arr: struc.AtomArray, ligand_resname: str, radius: float = 10.0) -> dict:
    """Residues with any atom within `radius` of the single drug-like ligand copy.
    Returns {ca:(N,3), cb:(N,3), restype:(N,), resid:(N,)} sorted by resid. Cβ=CA for Gly.
    Raises ValueError if the ligand is absent or no residues fall in the pocket."""
    het = arr[~struc.filter_amino_acids(arr)]
    lig = het[het.res_name == ligand_resname.upper()]
    if len(lig) == 0:
        raise ValueError(f"ligand {ligand_resname} not found")
    cid = str(lig.chain_id[0]); rid0 = int(lig.res_id[0])
    lig = lig[(lig.chain_id == cid) & (lig.res_id == rid0)]           # single physical copy

    prot = arr[struc.filter_amino_acids(arr)]
    if len(prot) == 0:
        raise ValueError("no protein atoms")
    dmin, _ = cKDTree(lig.coord).query(prot.coord, k=1)
    near = dmin <= radius
    near_keys = set(zip(prot.chain_id[near].tolist(), prot.res_id[near].tolist()))
    if not near_keys:
        raise ValueError("no residues within radius")

    ca, cb, rt, rid = [], [], [], []
    seen = set()
    for i in range(len(prot)):
        key = (prot.chain_id[i], int(prot.res_id[i]))
        if key not in near_keys or key in seen:
            continue
        rn = str(prot.res_name[i])
        if rn not in AA3_TO_IDX:
            continue
        seen.add(key)
        res = prot[(prot.chain_id == key[0]) & (prot.res_id == key[1])]
        ca_a = res[res.atom_name == "CA"].coord
        cb_a = res[res.atom_name == "CB"].coord
        ca_xyz = ca_a[0] if len(ca_a) else res.coord.mean(0)
        cb_xyz = cb_a[0] if len(cb_a) else ca_xyz
        ca.append(ca_xyz); cb.append(cb_xyz); rt.append(AA3_TO_IDX[rn]); rid.append(key[1])
    if not rt:
        raise ValueError("no standard-AA residues in pocket")
    order = np.argsort(rid)
    return {
        "ca": np.asarray(ca, dtype=np.float32)[order],
        "cb": np.asarray(cb, dtype=np.float32)[order],
        "restype": np.asarray(rt, dtype=np.int64)[order],
        "resid": np.asarray(rid, dtype=np.int64)[order],
    }
