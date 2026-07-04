"""Negative controls for the predicted-affinity result.

Scores the real generations vs four controls under one DTI scorer to test whether
the "beats best known ligand" rate is target-specific or a scorer artifact.

Controls:
  real          - the model's generations for each protein (the ~70.6% set)
  A_mismatch    - real molecules scored against a PERMUTED protein (derangement)
  B_foreign     - known ligands of OTHER targets, scored against this protein
  C_notrain     - the untrained-baseline generations
  D_random_real - random Enamine building-block molecules (no conditioning)
"""
import os
import pickle

import click
import numpy as np
import pandas as pd

from synformer.eval.affinity import load_scorer, predict_affinity


def derangement(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or all(p[i] != i for i in range(n)):
            return list(p)


def foreign_ligands(target, gt_by_target, r, seed):
    rng = np.random.default_rng(seed)
    own = set(gt_by_target.get(target, []))
    pool = sorted({s for t, smis in gt_by_target.items() if t != target for s in smis} - own)
    if not pool:
        return []
    return list(rng.choice(pool, size=min(r, len(pool)), replace=False))


def _best(model, smiles, seq):
    if not smiles:
        return float("nan")
    return float(predict_affinity(model, list(smiles), [seq] * len(smiles)).max())


@click.command()
@click.option("--real", "real_path", default="data/evaluations/epoch=23-step=28076/infos_2025-06-11_09-12-36.pkl")
@click.option("--notrain", "notrain_path", default="data/evaluations/notrain/infos.pkl")
@click.option("--pairs", default="/Users/micheldumontier/code/prot2drug/data/papyrus/papyrus_selection_182129.csv")
@click.option("--seqs", default="data/other/aa_seq_test.csv")
@click.option("--fpindex", "fpindex_path", default="data/processed/comp_2048/fpindex.pkl")
@click.option("--scorer", default="MPNN_CNN_DAVIS")
@click.option("--r", default=44, type=int, help="control molecules per protein")
@click.option("--seed", default=42, type=int)
@click.option("--out", default="data/evaluations/affinity_controls.csv")
def main(real_path, notrain_path, pairs, seqs, fpindex_path, scorer, r, seed, out):
    real = pickle.load(open(real_path, "rb"))
    notrain = pickle.load(open(notrain_path, "rb")) if os.path.exists(notrain_path) else {}
    seq_map = pd.read_csv(seqs).dropna(subset=["aa_seq"]).set_index("target_id")["aa_seq"].to_dict()
    gt = pd.read_csv(pairs).groupby("target_id")["SMILES"].apply(lambda s: sorted(set(s))).to_dict()
    fpindex = pickle.load(open(fpindex_path, "rb"))
    bb = [m.smiles for m in fpindex.molecules]
    rng = np.random.default_rng(seed)

    targets = [t for t in real if t in seq_map and t in gt]
    real_gen = {t: sorted({p["smiles"] for p in real[t].values()}) for t in targets}
    notrain_gen = {t: sorted({p["smiles"] for p in notrain.get(t, {}).values()}) for t in targets}
    perm = derangement(len(targets), seed)
    shuffled_seq = {t: seq_map[targets[perm[i]]] for i, t in enumerate(targets)}

    model = load_scorer(scorer)
    true_best = {t: _best(model, gt[t], seq_map[t]) for t in targets}

    conditions = {
        "real": lambda t: (real_gen[t], seq_map[t]),
        "A_mismatch": lambda t: (real_gen[t], shuffled_seq[t]),
        "B_foreign": lambda t: (foreign_ligands(t, gt, r, seed), seq_map[t]),
        "C_notrain": lambda t: (notrain_gen.get(t, []), seq_map[t]),
        "D_random_real": lambda t: (list(rng.choice(bb, size=min(r, len(bb)), replace=False)), seq_map[t]),
    }

    rows = []
    for name, fn in conditions.items():
        beats, cond_bests = [], []
        for t in targets:
            mols, seq = fn(t)
            if not mols or np.isnan(true_best[t]):
                continue
            cb = _best(model, mols, seq)
            cond_bests.append(cb)
            beats.append(cb >= true_best[t])
        rows.append({
            "condition": name,
            "n_proteins": len(cond_bests),
            "mean_cond_best": float(np.mean(cond_bests)) if cond_bests else float("nan"),
            "mean_true_best": float(np.nanmean([true_best[t] for t in targets])),
            "pct_beats_best": float(np.mean(beats)) * 100 if beats else float("nan"),
        })
        print(rows[-1])

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
