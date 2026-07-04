"""Predicted binding-affinity comparison: generated vs ground-truth ligands.

Closes the study's biggest gap. For each protein (with a known sequence), scores
its generated molecules and its known ligands with a pretrained DTI model
(DeepPurpose MPNN_CNN_DAVIS; higher = stronger predicted binding, pKd), then asks:
do generated molecules bind as well as the real ones?

Usage:
    python scripts/eval_affinity.py \
        --infos data/evaluations/epoch=23-step=28076/infos_2025-06-11_09-12-36.pkl \
        --pairs /path/to/papyrus_selection_182129.csv \
        --seqs data/other/aa_seq_test.csv \
        --out data/evaluations/epoch=23-step=28076/affinity.csv
"""
import pickle

import click
import numpy as np
import pandas as pd

from synformer.eval.affinity import load_scorer, predict_affinity


@click.command()
@click.option("--infos", "infos_path", required=True, type=click.Path(exists=True))
@click.option("--pairs", "pairs_path", required=True, type=click.Path(exists=True))
@click.option("--seqs", "seqs_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_path", required=True, type=click.Path())
def main(infos_path: str, pairs_path: str, seqs_path: str, out_path: str) -> None:
    infos = pickle.load(open(infos_path, "rb"))
    seqs = pd.read_csv(seqs_path).dropna(subset=["aa_seq"]).set_index("target_id")["aa_seq"].to_dict()
    pairs = pd.read_csv(pairs_path)
    true_by_target = pairs.groupby("target_id")["SMILES"].apply(lambda s: sorted(set(s))).to_dict()

    model = load_scorer("MPNN_CNN_DAVIS")
    rows = []
    targets = [t for t in infos if t in seqs and t in true_by_target]
    print(f"scoring {len(targets)} proteins")
    for i, t in enumerate(targets, 1):
        seq = seqs[t]
        gen = sorted({p["smiles"] for p in infos[t].values()})
        tru = true_by_target[t]
        try:
            gen_aff = predict_affinity(model, gen, [seq] * len(gen)) if gen else np.array([])
            tru_aff = predict_affinity(model, tru, [seq] * len(tru)) if tru else np.array([])
        except Exception as e:  # a bad SMILES / featurization failure for this target
            print(f"  [skip {t}] {type(e).__name__}: {e}")
            continue
        if not len(gen_aff) or not len(tru_aff):
            continue
        rows.append({
            "target_id": t,
            "n_gen": len(gen_aff), "n_true": len(tru_aff),
            "gen_mean": float(gen_aff.mean()), "gen_best": float(gen_aff.max()),
            "true_mean": float(tru_aff.mean()), "true_best": float(tru_aff.max()),
            "frac_gen_beats_true_best": float((gen_aff > tru_aff.max()).mean()),
        })
        if i % 25 == 0:
            print(f"  {i}/{len(targets)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nsaved {out_path} ({len(df)} proteins)")
    print("\n== Predicted binding affinity: generated vs ground-truth (pKd, higher=stronger) ==")
    print(f"mean over proteins  gen_best={df['gen_best'].mean():.3f}  true_best={df['true_best'].mean():.3f}")
    print(f"mean over proteins  gen_mean={df['gen_mean'].mean():.3f}  true_mean={df['true_mean'].mean():.3f}")
    print(f"proteins where best generated >= best known ligand: "
          f"{(df['gen_best'] >= df['true_best']).mean()*100:.1f}%")
    print(f"mean fraction of generated molecules beating the best known ligand: "
          f"{df['frac_gen_beats_true_best'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
