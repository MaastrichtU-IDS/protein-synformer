"""Evaluation harness: compute generation, synthesizability, and (optionally)
affinity metrics for a saved `infos` pickle from `scripts/evaluate.py`.

Usage:
    python scripts/run_eval.py \
        --infos data/evaluations/epoch=23-step=28076/infos_2025-06-11_09-12-36.pkl \
        --reference /path/to/papyrus_selection_182129.csv --repeat 100
"""
import pickle

import click
import numpy as np
import pandas as pd

from synformer.chem.mol import Molecule
from synformer.eval import generation as gen
from synformer.eval import synthesizability as syn


@click.command()
@click.option("--infos", "infos_path", required=True, type=click.Path(exists=True))
@click.option("--reference", "reference_path", default=None, type=click.Path(exists=True),
              help="CSV with a SMILES column of known ligands, for novelty.")
@click.option("--repeat", default=100, type=int, help="Attempts per protein used at sampling time.")
def main(infos_path: str, reference_path: str | None, repeat: int) -> None:
    infos = pickle.load(open(infos_path, "rb"))
    all_smiles = gen.flatten_smiles(infos)
    all_mols = [Molecule(s) for s in all_smiles]
    routes = syn.route_lengths(infos)

    print(f"proteins: {len(infos)} | total valid generations: {len(all_smiles)}")
    print("\n== Generation quality ==")
    print(f"validity rate (valid/attempts) : {gen.validity_rate(infos, repeat):.3f}")
    print(f"uniqueness                     : {gen.uniqueness(all_smiles):.3f}")
    if reference_path:
        ref = pd.read_csv(reference_path)
        col = "SMILES" if "SMILES" in ref.columns else "smiles"
        print(f"novelty vs reference ligands    : {gen.novelty(all_smiles, ref[col]):.3f}")
    print(f"internal diversity (per-protein): {gen.per_target_internal_diversity(infos):.3f}")
    print(f"scaffold diversity              : {gen.scaffold_diversity(all_mols):.3f}")

    print("\n== Synthesizability ==")
    print(f"mean SA score (1 easy - 10 hard): {syn.mean_sa_score(all_mols):.3f}")
    r = np.array(routes)
    print(f"route length (reactions)        : mean={r.mean():.2f} median={np.median(r):.0f} "
          f"min={r.min()} max={r.max()}")


if __name__ == "__main__":
    main()
