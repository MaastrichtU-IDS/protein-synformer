"""DPO preference-pair builder: pure functions that turn per-molecule, per-pocket smina
docking scores into specificity scores and winner/loser SMILES pairs for DPO training.

Docking scores are smina scores in kcal/mol: MORE NEGATIVE = BETTER BINDING. A molecule that
binds its own pocket well and mismatch pockets poorly gets a NEGATIVE specificity score; a
promiscuous molecule (binds mismatch pockets about as well as its own, or better) gets a
POSITIVE specificity score. Winners for DPO are therefore the MOST NEGATIVE-specificity
molecules and losers are the MOST POSITIVE. Getting this backwards would silently train the
model to prefer promiscuous binders -- see tests/test_dpo_pairs.py for hand-checked numbers.

Orchestration (docking, sampling mismatch pockets, writing pairs_<target>.json) is a later,
non-pure ops step and is intentionally NOT implemented here.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


def per_molecule_specificity(scores_df: pd.DataFrame, target: str) -> dict[str, float]:
    """For source ``target``'s docked molecules, compute a per-molecule specificity score.

    ``scores_df`` has columns ``target, pocket, molecule, source, score`` (score = smina
    kcal/mol, more negative = better binding). Rows are filtered to ``target == target``, then
    pivoted into a molecules x pockets matrix (min score per molecule/pocket cell, i.e. the
    best of any duplicate docking runs into the same pocket). Each pocket COLUMN is
    z-normalized (nan-aware, ``(x - nanmean) / nanstd``) across all molecules docked into that
    pocket -- mirroring ``scripts/powered_analyze.py::_delta_win_from_matrix`` -- so the
    own-pocket cell and every mismatch-pocket cell sit on the same per-pocket scale.

    For each molecule with a finite own-pocket z (pocket == target) and at least one finite
    mismatch-pocket z (pocket != target)::

        spec(m) = z_own - mean(z_mismatch)

    Molecules missing the own cell, or missing ALL mismatch cells, are skipped entirely.

    Because lower score = better binding, a molecule that binds its own pocket well and
    mismatch pockets poorly has z_own << 0 and mean(z_mismatch) >> 0, so spec is NEGATIVE for
    a specific molecule and POSITIVE for a promiscuous one. MORE NEGATIVE = MORE SPECIFIC.
    """
    sub = scores_df[scores_df["target"] == target]
    if sub.empty:
        return {}

    matrix = sub.pivot_table(index="molecule", columns="pocket", values="score", aggfunc="min")
    if target not in matrix.columns:
        return {}

    M = matrix.to_numpy(dtype=float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = (M - mu) / sd

    own_col = matrix.columns.get_loc(target)
    mismatch_cols = [j for j in range(matrix.shape[1]) if j != own_col]

    spec: dict[str, float] = {}
    for i, mol in enumerate(matrix.index):
        own_z = Z[i, own_col]
        if not np.isfinite(own_z):
            continue
        mismatch_z = [Z[i, j] for j in mismatch_cols if np.isfinite(Z[i, j])]
        if not mismatch_z:
            continue
        spec[mol] = float(own_z - float(np.mean(mismatch_z)))
    return spec


def make_pairs(spec_by_smiles: dict[str, float], frac: float = 0.3) -> list[tuple[str, str]]:
    """Build (winner, loser) SMILES pairs for DPO from per-molecule specificity scores.

    Winners = MORE SPECIFIC molecules = MORE NEGATIVE spec. Losers = MORE PROMISCUOUS
    molecules = MORE POSITIVE spec (see ``per_molecule_specificity`` for why negative =
    specific). Molecules are sorted by spec ascending; the lowest ``frac`` fraction becomes
    winners and the highest ``frac`` fraction becomes losers. Pairing is the full cross
    product of winners x losers (every winner paired with every loser) -- simple, and it
    maximizes the number of preference pairs extracted from a fixed pool of docked molecules.

    Returns [] if there are too few molecules to form a non-overlapping winner and loser set
    (k = floor(n * frac) < 1, or the winner/loser sets would overlap).
    """
    n = len(spec_by_smiles)
    k = math.floor(n * frac + 1e-9)  # + epsilon guards against float round-down (e.g. 10*0.3)
    if k < 1 or 2 * k > n:
        return []

    ordered = sorted(spec_by_smiles.items(), key=lambda kv: kv[1])
    winners = [smiles for smiles, _ in ordered[:k]]
    losers = [smiles for smiles, _ in ordered[-k:]]
    return list(itertools.product(winners, losers))
