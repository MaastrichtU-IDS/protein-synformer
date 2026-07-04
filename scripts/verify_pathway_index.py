"""Verify which fpindex the synthetic pathways were generated against.

Each pathway is a token program ([token_type, index] rows) whose REACTANT rows
index into a specific building-block ordering. Replaying the program with the
CORRECT fpindex reconstructs (a close analog of) the pathway's target molecule;
with the WRONG ordering it produces unrelated molecules. We replay a sample of
pathways through each candidate index and report mean similarity to the target.

Usage:
    python -m scripts.verify_pathway_index --n 300
"""
import pickle

import click
import numpy as np
import torch

from synformer.chem.mol import Molecule
from synformer.chem.stack import Stack
from synformer.data.common import TokenType

CANDIDATES = {
    "HF whgao/synformer": "data/processed/comp_hf",
    "rebuilt 2025 (comp_2048)": "data/processed/comp_2048",
}


def replay(pathway, fpindex, rxn_matrix):
    stack = Stack()
    arr = np.asarray(pathway)
    for tt, idx in arr:
        if tt == TokenType.START:
            continue
        if tt == TokenType.END:
            break
        if tt == TokenType.REACTION:
            if not stack.push_rxn(rxn_matrix.reactions[int(idx)], int(idx)):
                break
        elif tt == TokenType.REACTANT:
            stack.push_mol(fpindex.molecules[int(idx)], int(idx))
    if stack.get_stack_depth() >= 1:
        try:
            return stack.get_one_top()
        except Exception:
            return None
    return None


@click.command()
@click.option("--n", default=300, type=int)
def main(n):
    pw = torch.load("data/synthetic_pathways/filtered_pathways_370000.pth", map_location="cpu")
    items = list(pw.items())[:n]
    for label, path in CANDIDATES.items():
        try:
            fpindex = pickle.load(open(f"{path}/fpindex.pkl", "rb"))
            rxn_matrix = pickle.load(open(f"{path}/matrix.pkl", "rb"))
        except FileNotFoundError:
            print(f"{label:28s}: index not present, skipping")
            continue
        sims, exact, built = [], 0, 0
        for target_smiles, pathway in items:
            prod = replay(pathway, fpindex, rxn_matrix)
            if prod is None:
                continue
            built += 1
            s = Molecule(target_smiles).sim(prod)
            sims.append(s)
            if s >= 0.999:
                exact += 1
        sims = np.array(sims) if sims else np.array([0.0])
        print(f"{label:28s}: {len(fpindex.molecules)} mols | built {built}/{len(items)} | "
              f"mean sim {sims.mean():.3f} | exact {exact}/{built}")


if __name__ == "__main__":
    main()
