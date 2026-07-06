"""Smina docking wrapper.

Embeds a SMILES to 3D with RDKit, runs smina with autobox from a
co-crystal reference ligand, and parses the best affinity (mode 1).

Environment variable SMINA can override the default path.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem

from synformer.dock.receptor import ReceptorSpec

log = logging.getLogger(__name__)

# Path to smina binary; override with env var SMINA.
SMINA: str = os.environ.get(
    "SMINA",
    os.path.expanduser("~/miniforge3/envs/dock/bin/smina"),
)

# Default docking timeout in seconds.
DOCK_TIMEOUT: int = int(os.environ.get("DOCK_TIMEOUT", "600"))


def _embed(smiles: str, path: str) -> bool:
    """Embed a SMILES to a 3D SDF file via RDKit.

    Returns True on success, False on any failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        log.warning("RDKit could not parse SMILES: %s", smiles)
        return False

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        log.warning("EmbedMolecule failed for SMILES: %s", smiles)
        return False

    ff_result = AllChem.MMFFOptimizeMolecule(mol)
    if ff_result == -1:
        log.warning("MMFF force field could not be set up for SMILES: %s", smiles)
        # still usable — proceed with the unoptimised geometry

    Chem.MolToMolFile(mol, path)
    return True


def dock(spec: ReceptorSpec, smiles: str, seed: int = 0) -> float:
    """Dock a SMILES into the receptor described by *spec*.

    Parameters
    ----------
    spec:
        ReceptorSpec from prepare_target (receptor_path + ref_ligand_path).
    smiles:
        SMILES string of the molecule to dock.
    seed:
        Random seed passed to smina (reproducibility).

    Returns
    -------
    Best affinity in kcal/mol (lower = stronger binding).
    Returns ``float('nan')`` on embed failure or smina failure/timeout.
    """
    with tempfile.TemporaryDirectory() as d:
        lig_path = os.path.join(d, "lig.sdf")
        out_path = os.path.join(d, "out.sdf")

        if not _embed(smiles, lig_path):
            return float("nan")

        cmd = [
            SMINA,
            "--receptor", spec.receptor_path,
            "--ligand", lig_path,
            "--autobox_ligand", spec.ref_ligand_path,
            "--exhaustiveness", "8",
            "--seed", str(seed),
            "--out", out_path,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DOCK_TIMEOUT,
                check=False,  # parse output even on non-zero exit
            )
        except subprocess.TimeoutExpired:
            log.warning("smina timed out after %d s for SMILES: %s", DOCK_TIMEOUT, smiles)
            return float("nan")
        except FileNotFoundError:
            log.error("smina binary not found at: %s", SMINA)
            return float("nan")
        except Exception as exc:  # noqa: BLE001
            log.warning("smina subprocess error: %s", exc)
            return float("nan")

        stdout = proc.stdout
        # smina prints a score table to stdout; the first result row (mode 1) looks like:
        #    1      -7.4       0.000      0.000
        for line in stdout.splitlines():
            m = re.match(r"\s*1\s+(-?\d+\.\d+)", line)
            if m:
                affinity = float(m.group(1))
                log.info("smina best affinity: %.3f kcal/mol", affinity)
                return affinity

        # No mode-1 line found → smina may have failed silently
        log.warning(
            "smina produced no mode-1 affinity line.\nstdout:\n%s\nstderr:\n%s",
            stdout[:2000],
            proc.stderr[:500],
        )
        return float("nan")
