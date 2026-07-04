"""Reproduce the Fig-4 REAL-space coverage result.

SynFormer projects each ligand into synthesizable (Enamine REAL) space; an exact
match (projected == original) means the molecule is reachable in REAL space. The
paper reports ~20-22% exact matches. Operates on a projection-results CSV with
columns: lig_id, smiles_original, smiles_proj, similarity.

Usage:
    python scripts/reproduce_coverage.py --projections /path/to/synformer_ligands_test.csv
"""
import click
import pandas as pd

from synformer.chem.mol import Molecule


@click.command()
@click.option("--projections", "path", required=True, type=click.Path(exists=True))
def main(path: str) -> None:
    df = pd.read_csv(path)
    print(f"rows: {len(df)} | unique ligands: {df['lig_id'].nunique()} | "
          f"avg tries/ligand: {len(df) / df['lig_id'].nunique():.2f}")
    print(f"projection similarity: mean={df['similarity'].mean():.3f} "
          f"median={df['similarity'].median():.3f}")

    # Prefer the reported similarity; verify a sample against canonical SMILES equality.
    best = df.groupby("lig_id")["similarity"].max()
    exact = (best >= 0.999).mean()
    print(f"exact REAL-space coverage (best per ligand == 1.0): {exact * 100:.1f}%")
    print("report Fig-4: ~20% at 8 tries, ~22% at 16 tries")

    # Spot-check that similarity==1.0 really means identical canonical structure.
    hit = df[df["similarity"] >= 0.999].head(200)
    ok = sum(
        Molecule(str(r.smiles_original)).is_valid
        and Molecule(str(r.smiles_proj)).is_valid
        and Molecule(str(r.smiles_original)).csmiles == Molecule(str(r.smiles_proj)).csmiles
        for r in hit.itertuples()
    )
    print(f"canonical-equality check on {len(hit)} claimed exact matches: {ok}/{len(hit)} confirmed")


if __name__ == "__main__":
    main()
