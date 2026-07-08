"""AlphaFold receptor prep for the docking artifact arm: download the AF model for a
UniProt accession, superpose it onto the holo crystal receptor, and reuse the crystal
co-crystal ligand as the autobox reference (its coords land in the AF pocket after fit)."""
from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io

from synformer.dock.receptor import ReceptorSpec

log = logging.getLogger(__name__)

AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb"

# Correspondence-quality gate (see prepare_af_target): a "successful" homolog
# superposition can rest on as few as 3 coincidental anchors even for unrelated
# sequences, giving a deceptively small ca_rmsd computed only over that tiny
# anchor set. These thresholds are generous -- a real protein's own AF model vs.
# its crystal structure gives hundreds of anchors and sub-Å RMSD (e.g. RBP4 =
# 131 anchors / 0.34 Å) -- but they reject spurious few-anchor "fits".
MIN_ANCHORS_ABS = 20
MIN_ANCHORS_FRAC = 0.4
MAX_CA_RMSD = 5.0


@dataclass
class AFResult:
    spec: ReceptorSpec
    ca_rmsd: float
    pocket_plddt: float
    n_anchors: int


def _ca(atoms):
    return atoms[struc.filter_amino_acids(atoms) & (atoms.atom_name == "CA")]


def superpose_onto(fixed_ca, mobile_ca, mobile_full):
    """Superpose mobile onto fixed using homologous-CA correspondence.

    Returns the transformed FULL mobile structure, the CA-RMSD over matched
    residues, and the number of matched anchor residues (`n_anchors`) -- the
    latter is needed by callers to gate against spurious low-anchor "fits"
    (see `prepare_af_target`).
    """
    fitted_ca, transform, fixed_idx, mobile_idx = struc.superimpose_homologs(fixed_ca, mobile_ca)
    moved_full = transform.apply(mobile_full)
    # `fitted_ca` is a full-length copy of mobile_ca (transformed), not just the matched
    # anchor subset -- must index it by mobile_idx to line up with fixed_ca[fixed_idx]
    # for a valid per-atom RMSD (they can otherwise differ in length, e.g. a full-length
    # AF model vs. a crystal construct).
    rmsd = float(struc.rmsd(fixed_ca[fixed_idx], fitted_ca[mobile_idx]))
    n_anchors = len(fixed_idx)
    return moved_full, rmsd, n_anchors


def pocket_mean_plddt(moved_af, ref_ligand_path, radius: float = 8.0) -> float:
    """Mean AF b-factor (pLDDT) over RESIDUES with any atom within `radius` of the ref ligand.

    Averages ONE value per qualifying residue (its b-factor, which is constant across
    the residue's atoms for AF pLDDT), not per atom -- atom-averaging over-weights
    bulky residues (e.g. a 5-atom residue would count 5x as much as a 1-atom residue
    at the same pLDDT).

    NOTE: indexes the underlying coord/b_factor/res_id/chain_id arrays with a boolean
    mask rather than re-slicing the AtomArray itself (`moved_af[mask]`). biotite's
    AtomArray.__getitem__ rebuilds a fresh AtomArray from only its *registered*
    annotation categories (chain_id/res_id/.../element by default), so any attribute
    that isn't a registered annotation - e.g. b_factor set via plain
    `arr.b_factor = ...`, or read without `extra_fields=["b_factor"]` - would silently
    vanish on re-slicing.
    """
    lig = pdb_io.get_structure(pdb_io.PDBFile.read(ref_ligand_path), model=1)
    lig_xyz = lig.coord
    aa_mask = struc.filter_amino_acids(moved_af)
    coords = moved_af.coord[aa_mask]
    bfactors = moved_af.b_factor[aa_mask]
    chain_ids = moved_af.chain_id[aa_mask]
    res_ids = moved_af.res_id[aa_mask]

    atom_in_range = np.zeros(len(coords), dtype=bool)
    for i in range(len(coords)):
        if np.min(np.linalg.norm(lig_xyz - coords[i], axis=1)) <= radius:
            atom_in_range[i] = True

    if not atom_in_range.any():
        log.warning("pocket_mean_plddt: no residue within %.1f A of ligand %s", radius, ref_ligand_path)
        return float("nan")

    # Reduce atom-level hits to one row per residue (chain_id, res_id), then average
    # each residue's (constant) b-factor once.
    residue_keys = list(zip(chain_ids[atom_in_range], res_ids[atom_in_range]))
    seen = set()
    residue_plddts = []
    for key, bf in zip(residue_keys, bfactors[atom_in_range]):
        if key not in seen:
            seen.add(key)
            residue_plddts.append(bf)
    return float(np.mean(residue_plddts))


def prepare_af_target(accession: str, holo_spec: ReceptorSpec, out_dir: str) -> AFResult:
    """Download the AF model for `accession`, superpose it onto the holo crystal, and
    write an AF-based receptor that reuses the crystal's co-crystal ligand as the
    docking box.

    Parameters
    ----------
    accession:
        UniProt accession of the target (used to fetch its AlphaFold model).
    holo_spec:
        ReceptorSpec of the holo crystal receptor to superpose onto; its
        `ref_ligand_path` is reused unchanged as the AF receptor's autobox reference.
    out_dir:
        Directory in which to write `af_receptor.pdb` (created if absent).

    Returns
    -------
    AFResult with the AF-based ReceptorSpec, CA-RMSD and matched-anchor count from the
    superposition, and the per-residue pocket pLDDT.

    Raises
    ------
    ValueError
        If the homolog correspondence is too weak to trust (`n_anchors` too small
        relative to the shorter chain, or `ca_rmsd` too large) -- a "successful" fit
        from `superimpose_homologs` can rest on as few as 3 coincidental anchors even
        for unrelated sequences, which would otherwise silently produce a misaligned
        `af_receptor.pdb`. Callers running this over a batch of targets are expected
        to catch this and skip+log the pocket.
    """
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        af_path = os.path.join(tmp, f"AF-{accession}.pdb")
        urllib.request.urlretrieve(AF_URL.format(acc=accession), af_path)
        # extra_fields=["b_factor"] is required: biotite's get_structure omits b_factor
        # (pLDDT for AF models) by default, and pocket_mean_plddt needs it.
        af = pdb_io.get_structure(pdb_io.PDBFile.read(af_path), model=1, extra_fields=["b_factor"])
    holo = pdb_io.get_structure(pdb_io.PDBFile.read(holo_spec.receptor_path), model=1)

    fixed_ca, mobile_ca = _ca(holo), _ca(af)
    moved_af, rmsd, n_anchors = superpose_onto(fixed_ca, mobile_ca, af)
    log.info(
        "Superposed AF-%s onto %s: n_anchors=%d, ca_rmsd=%.2f A",
        accession, holo_spec.receptor_path, n_anchors, rmsd,
    )

    min_anchors_required = max(MIN_ANCHORS_ABS, MIN_ANCHORS_FRAC * min(len(fixed_ca), len(mobile_ca)))
    if n_anchors < min_anchors_required or rmsd > MAX_CA_RMSD:
        raise ValueError(
            f"AF-{accession}: untrustworthy correspondence to holo receptor "
            f"({n_anchors} anchors, need >= {min_anchors_required:.0f}; "
            f"ca_rmsd={rmsd:.2f} A, need <= {MAX_CA_RMSD} A) -- likely a spurious fit, skipping"
        )

    plddt = pocket_mean_plddt(moved_af, holo_spec.ref_ligand_path)
    log.info("AF-%s pocket mean pLDDT: %.1f", accession, plddt)

    af_receptor_path = os.path.join(out_dir, "af_receptor.pdb")
    prot = moved_af[struc.filter_amino_acids(moved_af)]
    f = pdb_io.PDBFile(); pdb_io.set_structure(f, prot); f.write(af_receptor_path)
    log.info("Wrote AF receptor: %s (%d atoms)", af_receptor_path, len(prot))

    return AFResult(
        spec=ReceptorSpec(receptor_path=af_receptor_path, ref_ligand_path=holo_spec.ref_ligand_path),
        ca_rmsd=rmsd, pocket_plddt=plddt, n_anchors=n_anchors)
