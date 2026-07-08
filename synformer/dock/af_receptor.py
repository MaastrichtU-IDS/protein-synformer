"""AlphaFold receptor prep for the docking artifact arm: download the AF model for a
UniProt accession, superpose it onto the holo crystal receptor, and reuse the crystal
co-crystal ligand as the autobox reference (its coords land in the AF pocket after fit)."""
from __future__ import annotations

import os
import tempfile
import urllib.request
from dataclasses import dataclass

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io

from synformer.dock.receptor import ReceptorSpec

AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb"


@dataclass
class AFResult:
    spec: ReceptorSpec
    ca_rmsd: float
    pocket_plddt: float


def _ca(atoms):
    return atoms[struc.filter_amino_acids(atoms) & (atoms.atom_name == "CA")]


def superpose_onto(fixed_ca, mobile_ca, mobile_full):
    """Superpose mobile onto fixed using homologous-CA correspondence; return the
    transformed FULL mobile structure and the CA-RMSD over matched residues."""
    fitted_ca, transform, fixed_idx, mobile_idx = struc.superimpose_homologs(fixed_ca, mobile_ca)
    moved_full = transform.apply(mobile_full)
    # `fitted_ca` is a full-length copy of mobile_ca (transformed), not just the matched
    # anchor subset -- must index it by mobile_idx to line up with fixed_ca[fixed_idx]
    # for a valid per-atom RMSD (they can otherwise differ in length, e.g. a full-length
    # AF model vs. a crystal construct).
    rmsd = float(struc.rmsd(fixed_ca[fixed_idx], fitted_ca[mobile_idx]))
    return moved_full, rmsd


def pocket_mean_plddt(moved_af, ref_ligand_path, radius: float = 8.0) -> float:
    """Mean AF b-factor (pLDDT) over residues with any atom within `radius` of the ref ligand.

    NOTE: indexes the underlying coord/b_factor arrays with a boolean mask rather than
    re-slicing the AtomArray itself (`moved_af[mask]`). biotite's AtomArray.__getitem__
    rebuilds a fresh AtomArray from only its *registered* annotation categories
    (chain_id/res_id/.../element by default), so any attribute that isn't a registered
    annotation - e.g. b_factor set via plain `arr.b_factor = ...`, or read without
    `extra_fields=["b_factor"]` - would silently vanish on re-slicing.
    """
    lig = pdb_io.get_structure(pdb_io.PDBFile.read(ref_ligand_path), model=1)
    lig_xyz = lig.coord
    aa_mask = struc.filter_amino_acids(moved_af)
    coords = moved_af.coord[aa_mask]
    bfactors = moved_af.b_factor[aa_mask]
    keep = np.zeros(len(coords), dtype=bool)
    for i in range(len(coords)):
        if np.min(np.linalg.norm(lig_xyz - coords[i], axis=1)) <= radius:
            keep[i] = True
    if not keep.any():
        return float("nan")
    return float(np.mean(bfactors[keep]))


def prepare_af_target(accession: str, holo_spec: ReceptorSpec, out_dir: str) -> AFResult:
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        af_path = os.path.join(tmp, f"AF-{accession}.pdb")
        urllib.request.urlretrieve(AF_URL.format(acc=accession), af_path)
        # extra_fields=["b_factor"] is required: biotite's get_structure omits b_factor
        # (pLDDT for AF models) by default, and pocket_mean_plddt needs it.
        af = pdb_io.get_structure(pdb_io.PDBFile.read(af_path), model=1, extra_fields=["b_factor"])
    holo = pdb_io.get_structure(pdb_io.PDBFile.read(holo_spec.receptor_path), model=1)
    moved_af, rmsd = superpose_onto(_ca(holo), _ca(af), af)
    plddt = pocket_mean_plddt(moved_af, holo_spec.ref_ligand_path)
    af_receptor_path = os.path.join(out_dir, "af_receptor.pdb")
    prot = moved_af[struc.filter_amino_acids(moved_af)]
    f = pdb_io.PDBFile(); pdb_io.set_structure(f, prot); f.write(af_receptor_path)
    return AFResult(
        spec=ReceptorSpec(receptor_path=af_receptor_path, ref_ligand_path=holo_spec.ref_ligand_path),
        ca_rmsd=rmsd, pocket_plddt=plddt)
