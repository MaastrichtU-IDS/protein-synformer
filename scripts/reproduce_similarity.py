"""Reproduce the Table III similarity statistics from a saved evaluation.

Recomputes max-Tanimoto (Morgan-4096) between generated molecules and
ground-truth ligands per protein, from a saved `infos` pickle (produced by
`scripts/evaluate.py`) and a protein-molecule-pairs CSV (columns: SMILES,
target_id). No model, GPU, or resampling required.

Usage:
    python scripts/reproduce_similarity.py \
        --infos data/evaluations/epoch=23-step=28076/infos_2025-06-11_09-12-36.pkl \
        --pairs /path/to/papyrus_selection_182129.csv
"""
import pickle

import click
import pandas as pd

from synformer.chem.mol import Molecule


def _stats(s: pd.Series) -> str:
    d = s.describe(percentiles=[0.25, 0.5, 0.75])
    return (f"mean={d['mean']:.4f} min={d['min']:.4f} 25%={d['25%']:.4f} "
            f"50%={d['50%']:.4f} 75%={d['75%']:.4f} max={d['max']:.4f}")


@click.command()
@click.option("--infos", "infos_path", required=True, type=click.Path(exists=True))
@click.option("--pairs", "pairs_path", required=True, type=click.Path(exists=True))
def main(infos_path: str, pairs_path: str) -> None:
    infos = pickle.load(open(infos_path, "rb"))
    gt = pd.read_csv(pairs_path)
    gt_by_target = gt.groupby("target_id")["SMILES"].apply(list).to_dict()

    covered = sum(1 for t in infos if t in gt_by_target)
    print(f"targets: {len(infos)} | ground-truth covered: {covered}/{len(infos)}")

    rows = []
    for tid, info in infos.items():
        gts = gt_by_target.get(tid)
        if not gts:
            continue
        gt_mols = [Molecule(s) for s in gts]
        for pred in info.values():
            mp = Molecule(pred["smiles"])
            for tsm, gm in zip(gts, gt_mols):
                rows.append((tid, tsm, pred["smiles"], mp.sim(gm)))

    df = pd.DataFrame(rows, columns=["target_id", "true_smiles", "pred_smiles", "similarity"]).drop_duplicates()
    print(f"pairwise similarity rows: {len(df)}")
    print("best per (protein,molecule) pair :", _stats(df.groupby(["target_id", "true_smiles"])["similarity"].max()))
    print("best per protein                 :", _stats(df.groupby(["target_id"])["similarity"].max()))
    print("all pairwise                     :", _stats(df["similarity"]))


if __name__ == "__main__":
    main()
