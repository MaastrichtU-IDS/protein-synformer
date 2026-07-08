"""Powered specificity analysis: bootstrap CIs for the crystal and AlphaFold docking
arms and their paired difference, plus Boltz discrimination AUROC at scale."""
from __future__ import annotations

import json

import click
import numpy as np
import pandas as pd

from scripts.boltz_controls_analyze import discrimination_auroc


def bootstrap_ci(values, stat, n_boot: int = 10000, seed: int = 42, alpha: float = 0.05):
    vals = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = [stat(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n_boot)]
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def _delta_win_from_matrix(M, target_ids):
    """Core of the clean normalized-delta metric, factored out of matrix construction so it
    is directly unit-testable on a synthetic matrix.

    ``M`` is an N×N array where ``M[i, j]`` is source ``i``'s best (min) docking score in
    pocket ``j`` (NaN if absent). Every column is full (every source was docked into every
    pocket), so — mirroring ``dock_analyze.py``'s trusted N=5 approach — we z-normalize
    within each COLUMN j (nan-aware, across all sources i), which puts the diagonal (own
    pocket) and every off-diagonal cell on exactly the same per-pocket scale. For each
    source i with a finite own (diagonal) cell and at least one finite off-diagonal z:
    ``delta_i = z(M[i,i]) - mean(z(M[i,j]) for j != i)``; ``win_i = 1.0 if delta_i < 0 else 0.0``.
    Targets whose own cell or all off-diagonal cells are NaN are skipped entirely.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = (M - mu) / sd
    delta, win = {}, {}
    for i, ti in enumerate(target_ids):
        own_z = Z[i, i]
        offs = [Z[i, j] for j in range(n) if j != i and np.isfinite(Z[i, j])]
        if not offs or not np.isfinite(own_z):
            continue
        delta[ti] = float(own_z - float(np.mean(offs)))
        win[ti] = 1.0 if delta[ti] < 0 else 0.0
    return delta, win


def _matrix_normalized_delta(scores_csv, target_ids, top_m_by_target):
    """Build the full N×N mismatch matrix from the scores CSV — M[i][j] = min score of
    source i's top-M SMILES (candidate source only) docked into pocket j, NaN if absent —
    then delegate to ``_delta_win_from_matrix`` for the per-column z-normalized delta/win.
    ``target_ids`` is used as BOTH the matrix rows and columns (every target is a pocket)."""
    df = pd.read_csv(scores_csv)
    df = df[df.source == "candidate"]
    n = len(target_ids)
    M = np.full((n, n), np.nan)
    for i, ti in enumerate(target_ids):
        mols = set(top_m_by_target.get(ti, []))
        if not mols:
            continue
        for j, tj in enumerate(target_ids):
            s = df[(df.molecule.isin(mols)) & (df.pocket == tj)].score.dropna()
            if len(s):
                M[i, j] = s.min()
    return _delta_win_from_matrix(M, target_ids)


def paired_diff_ci(a_by_t, b_by_t, seed: int = 42, n_boot: int = 10000):
    keys = [k for k in a_by_t if k in b_by_t]
    diffs = np.array([a_by_t[k] - b_by_t[k] for k in keys], dtype=float)
    lo, hi = bootstrap_ci(list(diffs), np.mean, n_boot=n_boot, seed=seed)
    return (float(np.nanmean(diffs)), lo, hi)


@click.command()
@click.option("--scores", default="data/dock/dock_scores.csv")
@click.option("--af-scores", default="data/dock/dock_scores_af.csv")
@click.option("--matrix", "matrix_json", default="data/dock/matrix_targets.json")
@click.option("--boltz-scores", default="data/boltz/boltz_controls_scores.csv")
@click.option("--n-candidates", default=150, type=int)
@click.option("--top-m", default=10, type=int)
@click.option("--out", default="data/dock/powered_specificity_summary.csv")
def main(scores, af_scores, matrix_json, boltz_scores, n_candidates, top_m, out):
    from scripts.dock_select import select_topm_for_target, _load_scores_table
    import pathlib
    target_ids = json.load(open(matrix_json))["targets"]
    # reconstruct top-M per target from crystal own-pocket candidate scores
    tbl = _load_scores_table(pathlib.Path(scores))
    dfc = pd.read_csv(scores)
    top_m_by = {}
    for ti in target_ids:
        cand = dfc[(dfc.target == ti) & (dfc.pocket == ti) & (dfc.source == "candidate")]
        own = {r.molecule: r.score for r in cand.itertuples()}
        top_m_by[ti] = select_topm_for_target(own, top_m)

    cd, cw = _matrix_normalized_delta(scores, target_ids, top_m_by)
    ad, aw = _matrix_normalized_delta(af_scores, target_ids, top_m_by)

    rows = []
    for label, d, w in [("crystal", cd, cw), ("alphafold", ad, aw)]:
        deltas = list(d.values()); wins = list(w.values())
        dlo, dhi = bootstrap_ci(deltas, np.mean)
        wlo, whi = bootstrap_ci(wins, np.mean)
        rows.append({"arm": label, "n": len(deltas), "delta_mean": float(np.mean(deltas)),
                     "delta_lo": dlo, "delta_hi": dhi, "win_mean": float(np.mean(wins)),
                     "win_lo": wlo, "win_hi": whi})
        print(f"{label:9}: delta {np.mean(deltas):+.3f} [{dlo:+.3f},{dhi:+.3f}]  "
              f"win {np.mean(wins):.2f} [{wlo:.2f},{whi:.2f}]  (n={len(deltas)})")
    pm, plo, phi = paired_diff_ci(cd, ad)
    print(f"crystal - AF paired delta-diff: {pm:+.3f} [{plo:+.3f},{phi:+.3f}]  "
          f"(<0 & CI excludes 0 => crystal more specific than AF => artifact-leaning)")
    rows.append({"arm": "crystal_minus_af", "n": len([k for k in cd if k in ad]),
                 "delta_mean": pm, "delta_lo": plo, "delta_hi": phi,
                 "win_mean": float("nan"), "win_lo": float("nan"), "win_hi": float("nan")})

    # Boltz discrimination at scale (pooled AUROC + CI over targets)
    b = pd.read_csv(boltz_scores)
    aurocs = []
    for ti in target_ids:
        sub = b[b.target == ti]
        a = discrimination_auroc(sub, "affinity_pred", higher_is_better=False)
        if a is not None:
            aurocs.append(a)
    if aurocs:
        alo, ahi = bootstrap_ci(aurocs, np.mean)
        print(f"boltz    : per-target AUROC(aff) mean {np.mean(aurocs):.3f} [{alo:.3f},{ahi:.3f}] "
              f"(n={len(aurocs)}; >0.5 => competent)")
        rows.append({"arm": "boltz_auroc", "n": len(aurocs), "delta_mean": float(np.mean(aurocs)),
                     "delta_lo": alo, "delta_hi": ahi, "win_mean": float("nan"),
                     "win_lo": float("nan"), "win_hi": float("nan")})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
