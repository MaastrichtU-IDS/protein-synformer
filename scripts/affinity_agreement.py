"""Second-scorer check: is the protein-blindness artifact general across DTI proxies?

Re-runs the decisive control (real vs mismatched-protein) under a second, independently
trained DTI model, and reports the per-protein rank correlation (Spearman) between the two
scorers' best-generated affinities. If scorer #2 also gives real == mismatch, the artifact
is general to DTI proxies (not a DeepPurpose-DAVIS quirk).
"""
import pickle

import click
import numpy as np
import pandas as pd

from scripts.affinity_controls import derangement
from synformer.eval.affinity import load_scorer, predict_affinity


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = x.argsort().argsort().astype(float), y.argsort().argsort().astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def _best(model, smiles, seq):
    if not smiles:
        return float("nan")
    return float(predict_affinity(model, list(smiles), [seq] * len(smiles)).max())


@click.command()
@click.option("--real", "real_path", default="data/evaluations/epoch=23-step=28076/infos_2025-06-11_09-12-36.pkl")
@click.option("--pairs", default="/Users/micheldumontier/code/prot2drug/data/papyrus/papyrus_selection_182129.csv")
@click.option("--seqs", default="data/other/aa_seq_test.csv")
@click.option("--scorer1", default="MPNN_CNN_DAVIS")
@click.option("--scorer2", default="MPNN_CNN_BindingDB")
@click.option("--seed", default=42, type=int)
@click.option("--out", default="data/evaluations/affinity_agreement.csv")
def main(real_path, pairs, seqs, scorer1, scorer2, seed, out):
    real = pickle.load(open(real_path, "rb"))
    seq_map = pd.read_csv(seqs).dropna(subset=["aa_seq"]).set_index("target_id")["aa_seq"].to_dict()
    gt = pd.read_csv(pairs).groupby("target_id")["SMILES"].apply(lambda s: sorted(set(s))).to_dict()

    targets = [t for t in real if t in seq_map and t in gt]
    real_gen = {t: sorted({p["smiles"] for p in real[t].values()}) for t in targets}
    perm = derangement(len(targets), seed)  # same seed as the controls run
    shuffled_seq = {t: seq_map[targets[perm[i]]] for i, t in enumerate(targets)}

    rows = {}
    real_best_by_scorer = {}
    for name in [scorer1, scorer2]:
        model = load_scorer(name)
        true_best = {t: _best(model, gt[t], seq_map[t]) for t in targets}
        real_best = {t: _best(model, real_gen[t], seq_map[t]) for t in targets}
        mis_best = {t: _best(model, real_gen[t], shuffled_seq[t]) for t in targets}
        valid = [t for t in targets if not np.isnan(true_best[t])]
        rows[name] = {
            "scorer": name,
            "pct_beats_real": float(np.mean([real_best[t] >= true_best[t] for t in valid])) * 100,
            "pct_beats_mismatch": float(np.mean([mis_best[t] >= true_best[t] for t in valid])) * 100,
            "mean_real_best": float(np.nanmean([real_best[t] for t in valid])),
            "mean_true_best": float(np.nanmean([true_best[t] for t in valid])),
        }
        real_best_by_scorer[name] = [real_best[t] for t in targets]
        print(rows[name])

    common = [i for i in range(len(targets))
              if not np.isnan(real_best_by_scorer[scorer1][i]) and not np.isnan(real_best_by_scorer[scorer2][i])]
    rho = spearman([real_best_by_scorer[scorer1][i] for i in common],
                   [real_best_by_scorer[scorer2][i] for i in common])
    print(f"per-protein Spearman(best-generated affinity) between {scorer1} and {scorer2}: {rho:.3f}  (n={len(common)})")

    pd.DataFrame(list(rows.values())).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
