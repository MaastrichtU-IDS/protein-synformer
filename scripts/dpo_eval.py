"""Held-out eval metric for the SP-DPO pilot: does the DPO-fine-tuned generator's raw
sample pool prefer its OWN (held-out) pocket over the ten TRAIN (mismatch) pockets more
than the base model's raw sample pool does?

This deliberately does NOT reuse ``scripts/powered_analyze.py::_delta_win_from_matrix``.
That helper z-normalizes WITHIN one pool (one source's row of the matrix), which forces
the pool-mean delta to ~0 by construction -- a false null when used to compare two pools
against each other -- and it operates on a pre-selected top-M subset, which reintroduces
docking-based SELECTION on top of whatever the generator produced. Here we are scoring the
GENERATOR (its raw, unselected sample pool), not a downstream selection step, so both
``own_preference`` and ``joint_z_own_preference`` below score every requested molecule on
a SHARED panel (the held-out target pocket + the fixed set of mismatch/train pockets) with
no within-pool z-normalization and no top-M filtering.

Two independent metrics are provided:

* ``own_preference`` -- the primary metric. A raw-score contrast (mean mismatch score minus
  own score) with an intuitive sign: d > 0 means "prefers its own pocket". Feed the DPO
  pool's d-values and the base pool's d-values to ``two_sample_diff_ci`` to test whether DPO
  shifts d upward relative to base.
* ``joint_z_own_preference`` -- a robustness check that instead z-normalizes scores within
  each pocket column across the POOLED base+DPO candidates (mirroring the project's
  existing z-delta convention in ``dpo_pairs.py``/``powered_analyze.py``), then splits the
  result back out by pool of origin. Its sign is the OPPOSITE of ``own_preference``'s: more
  NEGATIVE dz means more own-pocket-specific.
"""
from __future__ import annotations

import json

import click
import numpy as np
import pandas as pd


def own_preference(
    scores_df: pd.DataFrame,
    target: str,
    mismatch_pockets: list[str],
    smiles_subset=None,
) -> dict[str, float]:
    """Per-molecule own-pocket preference on a SHARED panel (no within-pool z, no top-M).

    ``scores_df`` has columns ``target, pocket, molecule, source, score`` (smina score,
    MORE NEGATIVE = BETTER BINDING). Rows are filtered to ``source == "candidate"``; if
    ``smiles_subset`` is given (a set/list of SMILES), rows are further filtered to only
    those molecules -- this is how the same scores frame is sliced into the base pool and
    the DPO pool for the eval CLI below.

    For each remaining molecule: ``own_score`` is its best (min) score among rows where
    ``pocket == target``; ``mismatch_scores`` is, for each pocket in ``mismatch_pockets``
    (which must NOT include ``target``), that pocket's best (min) score for the molecule
    (pockets the molecule was never docked into, or whose score is NaN, simply contribute
    no value). Then::

        d(m) = mean(mismatch_scores) - own_score

    SIGN: lower (more negative) score = better binding. A molecule that binds its own
    pocket well (own_score very negative) and the mismatch pockets worse (higher/less
    negative mismatch_scores) gets d(m) > 0 -- it PREFERS ITS OWN POCKET. Higher d = more
    own-preferring; d ~= 0 means no preference (promiscuous); d < 0 would mean it actually
    binds the mismatch pockets better than its own.

    Molecules with no finite own score, or no finite mismatch score in ANY mismatch
    pocket, are skipped entirely (nan-aware).
    """
    df = scores_df[scores_df["source"] == "candidate"]
    if smiles_subset is not None:
        subset = set(smiles_subset)
        df = df[df["molecule"].isin(subset)]

    d: dict[str, float] = {}
    for mol, g in df.groupby("molecule"):
        own_scores = g.loc[g["pocket"] == target, "score"].dropna()
        if own_scores.empty:
            continue
        own_score = float(own_scores.min())

        mismatch_scores = []
        for pocket in mismatch_pockets:
            s = g.loc[g["pocket"] == pocket, "score"].dropna()
            if not s.empty:
                mismatch_scores.append(float(s.min()))
        if not mismatch_scores:
            continue

        d[mol] = float(np.mean(mismatch_scores) - own_score)
    return d


def two_sample_diff_ci(
    a_values,
    b_values,
    seed: int = 42,
    n_boot: int = 10000,
    alpha: float = 0.05,
):
    """Unpaired bootstrap CI for ``mean(a) - mean(b)``.

    ``a`` and ``b`` need not be the same length or paired in any way (e.g. different
    numbers of molecules survived docking/filtering in each pool) -- each bootstrap
    iteration resamples ``a`` and ``b`` INDEPENDENTLY with replacement and recomputes the
    difference of means. Returns ``(diff, lo, hi)`` where ``diff`` is the observed
    (non-bootstrapped) ``mean(a) - mean(b)`` and ``(lo, hi)`` is the
    ``(alpha/2, 1 - alpha/2)`` percentile CI of the bootstrap distribution. Deterministic
    for a fixed ``seed`` via ``np.random.default_rng(seed)``.
    """
    a = np.asarray([v for v in a_values if v == v], dtype=float)  # drop NaN
    b = np.asarray([v for v in b_values if v == v], dtype=float)
    diff = float(np.mean(a) - np.mean(b))

    rng = np.random.default_rng(seed)
    a_boot = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    b_boot = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    boot_diffs = a_boot - b_boot

    lo = float(np.quantile(boot_diffs, alpha / 2))
    hi = float(np.quantile(boot_diffs, 1 - alpha / 2))
    return diff, lo, hi


def joint_z_own_preference(
    scores_df: pd.DataFrame,
    target: str,
    mismatch_pockets: list[str],
    origin_by_smiles: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Robustness variant of ``own_preference``: pocket-column z-normalization across the
    POOLED base+DPO candidates, then split back out by pool of origin.

    ``origin_by_smiles`` maps each molecule's SMILES to ``"base"`` or ``"dpo"`` and also
    defines the shared panel of molecules considered (only molecules present as keys are
    used). Rows are filtered to ``source == "candidate"``. The molecules x pockets matrix
    (``target`` plus every pocket in ``mismatch_pockets``, best/min score per cell) is
    z-normalized PER POCKET COLUMN (nan-aware ``(x - nanmean) / nanstd``) across ALL
    candidate molecules in the frame together (base and DPO jointly) -- mirroring
    ``scripts/dpo_pairs.py::per_molecule_specificity`` and
    ``scripts/powered_analyze.py::_delta_win_from_matrix``. For each molecule with a
    finite own-pocket z and at least one finite mismatch-pocket z::

        dz(m) = z(own) - mean(z(mismatch_pockets))

    SIGN (OPPOSITE of ``own_preference``'s raw ``d``): more NEGATIVE dz means MORE
    own-pocket-specific (own z-score is far below the mismatch z-scores), matching this
    project's existing z-delta convention. Molecules missing the own cell or all mismatch
    cells are skipped entirely.

    Returns ``{"base": {smiles: dz}, "dpo": {smiles: dz}}``.
    """
    pockets = [target] + list(mismatch_pockets)

    df = scores_df[scores_df["source"] == "candidate"]
    df = df[df["molecule"].isin(origin_by_smiles.keys())]
    df = df[df["pocket"].isin(pockets)]

    matrix = df.pivot_table(index="molecule", columns="pocket", values="score", aggfunc="min")
    matrix = matrix.reindex(columns=pockets)

    M = matrix.to_numpy(dtype=float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = (M - mu) / sd

    own_col = 0
    mismatch_cols = list(range(1, len(pockets)))

    result: dict[str, dict[str, float]] = {"base": {}, "dpo": {}}
    for i, mol in enumerate(matrix.index):
        own_z = Z[i, own_col]
        if not np.isfinite(own_z):
            continue
        mismatch_z = [Z[i, j] for j in mismatch_cols if np.isfinite(Z[i, j])]
        if not mismatch_z:
            continue
        dz = float(own_z - float(np.mean(mismatch_z)))
        origin = origin_by_smiles.get(mol)
        if origin in result:
            result[origin][mol] = dz
    return result


def _read_smi(path: str) -> list[str]:
    lines = [ln.strip() for ln in open(path)]
    return [ln for ln in lines if ln]


@click.command()
@click.option("--scores", required=True, help="Docking scores CSV (target,pocket,molecule,source,score).")
@click.option("--target", required=True, help="Held-out target/pocket id.")
@click.option("--mismatch-pockets", required=True, help="Comma-separated list of train pocket ids.")
@click.option("--base-smi", required=True, type=click.Path(exists=True), help="Base model's raw sample pool (.smi).")
@click.option("--dpo-smi", required=True, type=click.Path(exists=True), help="DPO model's raw sample pool (.smi).")
@click.option("--out", default=None, help="Optional path to write the JSON result.")
def main(scores, target, mismatch_pockets, base_smi, dpo_smi, out):
    mismatch = [p.strip() for p in mismatch_pockets.split(",") if p.strip()]
    df = pd.read_csv(scores)

    base_smiles = _read_smi(base_smi)
    dpo_smiles = _read_smi(dpo_smi)

    base_d = own_preference(df, target, mismatch, smiles_subset=base_smiles)
    dpo_d = own_preference(df, target, mismatch, smiles_subset=dpo_smiles)

    diff, lo, hi = two_sample_diff_ci(list(dpo_d.values()), list(base_d.values()))

    result = {
        "target": target,
        "n_base": len(base_d),
        "n_dpo": len(dpo_d),
        "mean_d_base": float(np.mean(list(base_d.values()))) if base_d else float("nan"),
        "mean_d_dpo": float(np.mean(list(dpo_d.values()))) if dpo_d else float("nan"),
        "diff": diff,
        "ci_lo": lo,
        "ci_hi": hi,
    }

    print(json.dumps(result, indent=2))
    if out:
        json.dump(result, open(out, "w"), indent=2)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
