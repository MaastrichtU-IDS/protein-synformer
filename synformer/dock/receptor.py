"""Holo receptor preparation for smina docking.

Fetches a PDB entry from RCSB, splits it into:
  - receptor.pdb  : protein ATOM records only
  - ref_ligand.pdb: HETATM records for the named co-crystal ligand

The ref_ligand PDB is used directly by smina's --autobox_ligand flag.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io

log = logging.getLogger(__name__)


@dataclass
class ReceptorSpec:
    """Paths produced by prepare_target."""

    receptor_path: str  # protein-only PDB
    ref_ligand_path: str  # co-crystal ligand PDB (for --autobox_ligand)


def prepare_target(
    pdb_id: str,
    out_dir: str,
    chain: str | None = None,
    ligand_resname: str | None = None,
) -> ReceptorSpec:
    """Fetch a holo PDB, write receptor.pdb and ref_ligand.pdb.

    Parameters
    ----------
    pdb_id:
        4-letter RCSB PDB accession (case-insensitive).
    out_dir:
        Directory in which to write the output files (created if absent).
    chain:
        If given, restrict the receptor to this chain ID.
    ligand_resname:
        3-letter residue name of the co-crystal ligand (e.g. "BTN").
        Required to write the ref_ligand file.

    Returns
    -------
    ReceptorSpec with absolute paths for the two output files.
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Fetch the PDB into a temp dir, then parse
    with tempfile.TemporaryDirectory() as tmp:
        fetched = rcsb.fetch(pdb_id.upper(), "pdb", target_path=tmp)
        pdb_file = pdb_io.PDBFile.read(fetched)
        # get_structure returns the first model by default (model=1)
        arr = pdb_io.get_structure(pdb_file, model=1)

    # --- Protein receptor ---
    prot_mask = struc.filter_amino_acids(arr)
    if chain is not None:
        prot_mask = prot_mask & (arr.chain_id == chain)
    receptor_atoms = arr[prot_mask]

    if len(receptor_atoms) == 0:
        raise ValueError(
            f"No protein atoms found in {pdb_id}"
            + (f" chain {chain}" if chain else "")
        )

    receptor_path = os.path.join(out_dir, "receptor.pdb")
    rec_pdb = pdb_io.PDBFile()
    pdb_io.set_structure(rec_pdb, receptor_atoms)
    rec_pdb.write(receptor_path)
    log.info("Wrote receptor: %s (%d atoms)", receptor_path, len(receptor_atoms))

    # --- Reference ligand ---
    if ligand_resname is None:
        raise ValueError("ligand_resname is required to write the ref_ligand file")

    lig_mask = arr.res_name == ligand_resname.upper()
    if chain is not None:
        lig_mask = lig_mask & (arr.chain_id == chain)
    lig_atoms = arr[lig_mask]

    if len(lig_atoms) == 0:
        raise ValueError(
            f"Ligand residue '{ligand_resname}' not found in {pdb_id}"
            + (f" chain {chain}" if chain else "")
        )

    # Use the first occurrence of this residue (lowest res_id)
    first_res_id = int(lig_atoms.res_id[0])
    lig_atoms = lig_atoms[lig_atoms.res_id == first_res_id]

    ref_ligand_path = os.path.join(out_dir, "ref_ligand.pdb")
    lig_pdb = pdb_io.PDBFile()
    pdb_io.set_structure(lig_pdb, lig_atoms)
    lig_pdb.write(ref_ligand_path)
    log.info("Wrote ref ligand: %s (%d atoms)", ref_ligand_path, len(lig_atoms))

    return ReceptorSpec(
        receptor_path=receptor_path,
        ref_ligand_path=ref_ligand_path,
    )
