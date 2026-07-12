"""Ops glue for the SP-DPO pilot: turn the docked scores into per-target winner/loser
preference pair files, using the unit-tested pure functions in ``scripts.dpo_pairs``.

    .venv/bin/python -m scripts.build_dpo_pairs \
        --scores data/dock/dpo/dpo_dock_scores.csv \
        --targets data/dock/dpo/train10.json \
        --out-dir data/dock/dpo/pairs

Writes ``<out-dir>/pairs_<target>.json`` (list of [winner_smiles, loser_smiles]) plus a
``summary.json`` (mols scored + pairs per target). No new logic — all specificity/pairing
math lives in scripts.dpo_pairs (tested).
"""
import json
from pathlib import Path

import click
import pandas as pd

from scripts.dpo_pairs import per_molecule_specificity, make_pairs


@click.command()
@click.option("--scores", default="data/dock/dpo/dpo_dock_scores.csv")
@click.option("--targets", default="data/dock/dpo/train10.json")
@click.option("--out-dir", default="data/dock/dpo/pairs")
@click.option("--frac", default=0.3, type=float, help="Top/bottom fraction for winners/losers.")
def main(scores, targets, out_dir, frac):
    df = pd.read_csv(scores)
    tids = [t["target_id"] for t in json.load(open(targets))]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    summary = {}
    for tid in tids:
        spec = per_molecule_specificity(df, tid)
        pairs = make_pairs(spec, frac=frac)
        json.dump(pairs, open(f"{out_dir}/pairs_{tid}.json", "w"))
        summary[tid] = {"n_mols_scored": len(spec), "n_pairs": len(pairs)}
        print(f"{tid}: {len(spec)} mols scored, {len(pairs)} pairs", flush=True)
    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=2)
    tot = sum(v["n_pairs"] for v in summary.values())
    print(f"TOTAL: {tot} pairs across {len(tids)} targets -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
