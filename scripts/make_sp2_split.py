"""Deterministic, protein-disjoint train/val/test split for SP2 fair comparison."""
import os

import click
import pandas as pd

SRC = "/Users/micheldumontier/code/prot2drug/data/papyrus/papyrus_selection_182129.csv"


def split_targets(target_ids):
    uniq = sorted(set(target_ids))
    train, val, test = set(), set(), set()
    for i, t in enumerate(uniq):
        (test if i % 20 == 0 else val if i % 20 == 1 else train).add(t)
    return train, val, test


@click.command()
@click.option("--src", default=SRC, type=click.Path(exists=True), help="Papyrus selection CSV with SMILES,target_id columns")
@click.option("--out-dir", default="data/protein_molecule_pairs", help="Directory to write sp2_{train,val,test}.csv")
def main(src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(src)
    train, val, test = split_targets(df["target_id"].tolist())
    for name, tset in [("train", train), ("val", val), ("test", test)]:
        sub = df[df["target_id"].isin(tset)]
        path = f"{out_dir}/sp2_{name}.csv"
        sub.to_csv(path, index=False)
        print(f"{path}: {len(sub)} pairs, {sub['target_id'].nunique()} proteins")


if __name__ == "__main__":
    main()
