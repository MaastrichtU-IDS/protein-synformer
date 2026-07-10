"""Analyze the Boltz-2 co-folding mismatch matrix and compare it to the docking matrix
built from the SAME top-1 hits. Reuses synformer.dock.geometry.mismatch_summary."""
from __future__ import annotations

import json

import click
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from synformer.dock.geometry import mismatch_summary


def build_matrix(scores_csv, target_ids, value_col, row_key, col_key):
    """n x n matrix; entry [i,j] = value_col where row_key==target_i and col_key==target_j.
    If multiple rows match a cell, take the min (best). NaN if none."""
    df = pd.read_csv(scores_csv)
    n = len(target_ids)
    M = np.full((n, n), np.nan)
    for i, ti in enumerate(target_ids):
        for j, tj in enumerate(target_ids):
            sub = df[(df[row_key] == ti) & (df[col_key] == tj)][value_col].dropna()
            if len(sub):
                M[i, j] = sub.min()
    return M


def normalized_summary(M):
    Z = (M - np.nanmean(M, axis=0, keepdims=True)) / np.nanstd(M, axis=0, keepdims=True)
    return mismatch_summary(Z)


def compare_matrices(A, B):
    a, b = [], []
    n = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not (np.isnan(A[i, j]) or np.isnan(B[i, j])):
                a.append(A[i, j]); b.append(B[i, j]); n += 1
    a, b = np.array(a), np.array(b)
    sp = float(spearmanr(a, b).statistic) if n >= 3 else float("nan")
    pe = float(pearsonr(a, b)[0]) if n >= 3 else float("nan")
    # sign agreement on own<offdiag per target
    def own_lt_off(M):
        out = []
        for i in range(M.shape[0]):
            off = np.nanmean([M[i, j] for j in range(M.shape[1]) if j != i])
            out.append(M[i, i] < off if not (np.isnan(M[i, i]) or np.isnan(off)) else None)
        return out
    oa, ob = own_lt_off(A), own_lt_off(B)
    pairs = [(x, y) for x, y in zip(oa, ob) if x is not None and y is not None]
    agree = float(np.mean([x == y for x, y in pairs])) if pairs else float("nan")
    return {"spearman": sp, "pearson": pe, "sign_agreement": agree, "n": n}


@click.command()
@click.option("--boltz-scores", default="data/boltz/boltz_scores.csv")
@click.option("--dock-scores", default="data/dock/dock_scores.csv")
@click.option("--inputs", default="data/boltz/matrix_inputs.json")
@click.option("--targets", default="data/dock/targets.json")
@click.option("--out", default="data/boltz/boltz_mismatch_summary.csv")
def main(boltz_scores, dock_scores, inputs, targets, out):
    target_ids = [t["target_id"] for t in json.load(open(targets))]

    # Boltz matrix on affinity_pred (row=hit_target, col=protein)
    B = build_matrix(boltz_scores, target_ids, "affinity_pred", "hit_target", "protein")

    # Docking matrix over the SAME top-1 hits: for each target, its hit's docking score per pocket.
    hit_of = {}
    for h in json.load(open(inputs))["hits"]:
        hit_of.setdefault(h["target_id"], h["smiles"])  # first (top-1) per target
    dock = pd.read_csv(dock_scores)
    n = len(target_ids)
    D = np.full((n, n), np.nan)
    for i, ti in enumerate(target_ids):
        smi = hit_of[ti]
        for j, tj in enumerate(target_ids):
            sub = dock[(dock.molecule == smi) & (dock.pocket == tj)]["score"].dropna()
            if len(sub):
                D[i, j] = sub.min()

    braw = mismatch_summary(B); bnorm = normalized_summary(B)
    draw = mismatch_summary(D); dnorm = normalized_summary(D)
    cmp_raw = compare_matrices(B, D)

    print("=== Boltz matrix (affinity_pred) ===")
    hdr = "hit\\prot"
    print(f"{hdr:12}" + "".join(f"{t:>10}" for t in target_ids))
    for i, ti in enumerate(target_ids):
        print(f"{ti:12}" + "".join(f"{B[i,j]:10.2f}" for j in range(n)))
    print(f"\nBOLTZ  raw : delta={braw['delta']:.3f} win={braw['win_rate']:.2f}")
    print(f"BOLTZ  norm: delta={bnorm['delta']:.3f} win={bnorm['win_rate']:.2f}")
    print(f"DOCK   raw : delta={draw['delta']:.3f} win={draw['win_rate']:.2f}")
    print(f"DOCK   norm: delta={dnorm['delta']:.3f} win={dnorm['win_rate']:.2f}")
    print(f"COMPARE    : spearman={cmp_raw['spearman']:.3f} pearson={cmp_raw['pearson']:.3f} "
          f"sign_agree={cmp_raw['sign_agreement']:.2f} n={cmp_raw['n']}")

    pd.DataFrame([
        {"method": "boltz", "view": "raw", **braw},
        {"method": "boltz", "view": "norm", **bnorm},
        {"method": "dock_same_hits", "view": "raw", **draw},
        {"method": "dock_same_hits", "view": "norm", **dnorm},
        {"method": "compare", "view": "boltz_vs_dock", **cmp_raw},
    ]).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
