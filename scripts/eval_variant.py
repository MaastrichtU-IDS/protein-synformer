"""Sample a trained variant on the SP2 test proteins and score it with the SP1 harness."""
import pickle

import click
import numpy as np
import pandas as pd
import torch

from scripts.sample_helpers import load_model, load_protein_embeddings, sample
from synformer.chem.mol import Molecule
from synformer.eval import generation as gen
from synformer.eval import synthesizability as syn


def summarize_infos(infos, gt_by_target, repeat):
    all_smiles = gen.flatten_smiles(infos)
    all_mols = [Molecule(s) for s in all_smiles]
    ref = [s for smis in gt_by_target.values() for s in smis]
    sims = []
    for t, info in infos.items():
        gts = gt_by_target.get(t)
        if not gts:
            continue
        gtm = [Molecule(s) for s in gts]
        for pred in info.values():
            mp = Molecule(pred["smiles"])
            best = max((mp.sim(g) for g in gtm), default=0.0)
            sims.append(best)
    return {
        "n_proteins": len(infos),
        "validity": gen.validity_rate(infos, repeat),
        "sim_best_pair_mean": float(np.mean(sims)) if sims else 0.0,
        "uniqueness": gen.uniqueness(all_smiles),
        "novelty": gen.novelty(all_smiles, ref),
        "internal_diversity": gen.per_target_internal_diversity(infos),
        "scaffold_diversity": gen.scaffold_diversity(all_mols),
        "mean_sa": syn.mean_sa_score(all_mols),
        "route_len_mean": float(np.mean(syn.route_lengths(infos))) if all_smiles else 0.0,
    }


@click.command()
@click.option("--variant", required=True)
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--pairs", default="data/protein_molecule_pairs/sp2_test.csv")
@click.option("--n-proteins", default=60, type=int)
@click.option("--repeat", default=60, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--device", default="mps")
@click.option("--out", default="data/evaluations/sp2_results.csv")
def main(variant, checkpoint, pairs, n_proteins, repeat, seed, device, out):
    torch.manual_seed(seed)
    model, fpindex, rxn_matrix = load_model(checkpoint, None, device)
    emb = load_protein_embeddings("data/protein_embeddings/embeddings_selection_float16_4973.pth")
    df = pd.read_csv(pairs)
    gt_by_target = df.groupby("target_id")["SMILES"].apply(lambda s: sorted(set(s))).to_dict()
    targets = [t for t in gt_by_target if t in emb][:n_proteins]

    infos = {}
    for i, t in enumerate(targets, 1):
        info, _ = sample(t, model, fpindex, rxn_matrix, emb, device, repeat=repeat)
        infos[t] = info
        if i % 10 == 0:
            print(f"  {i}/{len(targets)}")

    row = {"variant": variant, **summarize_infos(infos, gt_by_target, repeat)}
    print(row)
    hdr = not __import__("os").path.exists(out)
    pd.DataFrame([row]).to_csv(out, mode="a", header=hdr, index=False)
    print(f"appended to {out}")


if __name__ == "__main__":
    main()
