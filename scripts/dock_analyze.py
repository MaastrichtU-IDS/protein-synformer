"""Reproduce the docking-selection analysis from raw dock scores.

Recomputes, from ``data/dock/dock_scores.csv`` (the authoritative per-dock log):
  - Result 1: own-pocket top-M-selected vs all-candidate vs known vs random means.
  - Result 2 (raw): the M x M mismatch matrix + mismatch_summary (own/offdiag/delta/win_rate).
  - Result 2 (de-confounded): the SAME matrix column-normalised (z-score within each pocket, to
    remove per-pocket dockability bias) + a rank-based view (where each target's own molecules
    rank among all sources docked into that pocket).

The de-confounded view is the honest specificity metric: raw absolute docking scores are dominated
by which pocket is "easy" (binds everything), not by molecule-pocket match. See
docs/DOCKING_SELECTION_RESULTS.md.

Usage:
    ./.venv/bin/python -m scripts.dock_analyze \
        --scores data/dock/dock_scores.csv --targets data/dock/targets.json --top-m 10
"""

from __future__ import annotations

import json

import click
import numpy as np
import pandas as pd

from synformer.dock.geometry import mismatch_summary


def _top_m_smiles(own: pd.DataFrame, target: str, m: int) -> set[str]:
    """SMILES of the m best-scoring (lowest) own-pocket candidates for a target."""
    cand = own[(own.target == target) & (own.source == "candidate")]
    return set(cand.nsmallest(m, "score").molecule)


@click.command()
@click.option("--scores", default="data/dock/dock_scores.csv", show_default=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("--targets", default="data/dock/targets.json", show_default=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("--top-m", default=10, show_default=True, type=int)
def main(scores: str, targets: str, top_m: int) -> None:
    df = pd.read_csv(scores)
    target_ids = [t["target_id"] for t in json.load(open(targets))]
    # keep only targets that actually have own-pocket rows
    target_ids = [t for t in target_ids if not df[(df.target == t) & (df.pocket == t)].empty]
    own = df[df.target == df.pocket]

    # ---- Result 1: own-pocket source means -------------------------------------------------
    print("=== Result 1 — own-pocket means (kcal/mol, lower = stronger) ===")
    print(f"{'target':12} {'top-M sel':>10} {'all cand':>9} {'known':>9} {'random':>9}"
          f"  {'nkn':>3} {'nrnd':>4}")
    sel: dict[str, set[str]] = {}
    for t in target_ids:
        sel[t] = _top_m_smiles(own, t, top_m)
        c = own[(own.target == t) & (own.source == "candidate")].score
        topm = c.nsmallest(top_m)
        k = own[(own.target == t) & (own.source == "known")].score
        r = own[(own.target == t) & (own.source == "random")].score
        print(f"{t:12} {topm.mean():10.2f} {c.mean():9.2f} {k.mean():9.2f} {r.mean():9.2f}"
              f"  {len(k):3d} {len(r):4d}")

    # ---- Result 2: mismatch matrix ---------------------------------------------------------
    n = len(target_ids)
    M = np.full((n, n), np.nan)
    for i, ti in enumerate(target_ids):
        for j, tj in enumerate(target_ids):
            s = df[(df.molecule.isin(sel[ti])) & (df.pocket == tj)].score.dropna()
            if len(s):
                M[i, j] = s.min()

    print("\n=== Result 2 — mismatch matrix  M[i,j] = min(top-M of i, in pocket j) ===")
    print(f"{'src\\pocket':12}" + "".join(f"{t:>10}" for t in target_ids))
    for i, ti in enumerate(target_ids):
        print(f"{ti:12}" + "".join(f"{M[i, j]:10.2f}" for j in range(n)))

    raw = mismatch_summary(M)
    print(f"\nRAW        : own={raw['own_mean']:.3f}  offdiag={raw['offdiag_mean']:.3f}  "
          f"delta={raw['delta']:.3f}  win_rate={raw['win_rate']:.2f}")

    # ---- Result 2 de-confounded: column-normalise then rank --------------------------------
    Z = (M - M.mean(axis=0, keepdims=True)) / M.std(axis=0, keepdims=True)
    norm = mismatch_summary(Z)
    print(f"NORMALISED : own={norm['own_mean']:.3f}  offdiag={norm['offdiag_mean']:.3f}  "
          f"delta={norm['delta']:.3f}  win_rate={norm['win_rate']:.2f}  "
          "(z within each pocket — removes per-pocket dockability bias)")

    ranks = []
    print("\n=== own-source rank within each pocket (1 = best of all sources into that pocket) ===")
    for j, tj in enumerate(target_ids):
        order = list(np.argsort(M[:, j]))  # ascending: best first
        rank = order.index(j) + 1
        ranks.append(rank)
        print(f"  pocket {tj:12}: own ranks #{rank} of {n}")
    print(f"mean own-in-own rank = {np.mean(ranks):.2f}  (chance = {(n + 1) / 2:.1f}; lower = specific)")


if __name__ == "__main__":
    main()
