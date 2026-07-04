"""Synthesizability metrics.

Route length comes for free from the decoder: each saved generation records
`cnt_rxn` (number of reactions in its synthesis route). SA score uses RDKit's
contrib synthetic-accessibility scorer (1 = easy to make, 10 = hard).
"""
import os
import sys
from collections.abc import Sequence

import numpy as np
from rdkit.Chem import RDConfig

from synformer.chem.mol import Molecule

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402  (provided by RDKit contrib)


def sa_score(mol: Molecule) -> float:
    return sascorer.calculateScore(mol._rdmol)


def mean_sa_score(mols: Sequence[Molecule]) -> float:
    vals = [sa_score(m) for m in mols if m.is_valid]
    return float(np.mean(vals)) if vals else float("nan")


def route_lengths(infos: dict) -> list[int]:
    """Number of reactions per generated synthesis route, across all generations."""
    return [pred["cnt_rxn"] for info in infos.values() for pred in info.values()]
