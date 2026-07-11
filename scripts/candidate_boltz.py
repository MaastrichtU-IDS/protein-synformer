"""Stratified-sample SP-C pocket candidates across the smina range and Boltz-score them
(reusing boltz_controls._run_batch), for the candidate-regime scorer-agreement benchmark."""
from __future__ import annotations

import json
import pathlib

import click
import numpy as np
import pandas as pd


def load_candidates(pocket_scores_csv, targets) -> pd.DataFrame:
    d = pd.read_csv(pocket_scores_csv)
    d = d[(d.target.isin(targets)) & (d.pocket == d.target) & (d.source == "candidate")]
    return (d[["target", "molecule", "score"]].rename(columns={"score": "smina"})
            .dropna(subset=["smina"]).drop_duplicates(["target", "molecule"]))


def _pick_evenly(idx: list[int], count: int) -> list[int]:
    if count >= len(idx):
        return idx
    pos = np.linspace(0, len(idx) - 1, count).round().astype(int)
    return [idx[p] for p in sorted(set(pos.tolist()))]


def stratified_sample(df, n_per_target: int = 30, strata: int = 3) -> pd.DataFrame:
    per_stratum = n_per_target // strata
    out = []
    for target, g in df.groupby("target"):
        g = g.sort_values("smina").reset_index(drop=True)
        bins = np.array_split(np.arange(len(g)), strata)
        chosen: list[int] = []
        for b in bins:
            chosen += _pick_evenly(list(b), per_stratum)
        out.append(g.iloc[sorted(set(chosen))])
    return pd.concat(out, ignore_index=True)


@click.command()
@click.option("--pocket-scores", default="data/dock/dock_scores_pocket.csv")
@click.option("--inputs", default="data/boltz/matrix_inputs_powered.json")
@click.option("--targets", default="O43570_WT,P06537_WT,P10721_WT,P02753_WT,P0C559_WT")
@click.option("--n", default=30, type=int)
@click.option("--scores", default="data/dock/sp_cc_candidate_boltz.csv")
@click.option("--out-dir", default="boltz_out/sp_cc")
@click.option("--batch-in", default="boltz_batch_in_sp_cc")
@click.option("--samples", default=3, type=int)
def main(pocket_scores, inputs, targets, n, scores, out_dir, batch_in, samples):
    from scripts.boltz_controls import _run_batch, stem_for

    tlist = [t.strip() for t in targets.split(",")]
    seq_of = {p["target_id"]: p["sequence"] for p in json.load(open(inputs))["proteins"]}
    sample = stratified_sample(load_candidates(pocket_scores, tlist), n_per_target=n)
    cells = [{"target": r.target, "class": "candidate", "smiles": r.molecule,
              "sequence": seq_of[r.target], "stem": stem_for(r.target, "candidate", r.molecule)}
             for r in sample.itertuples() if r.target in seq_of]
    pathlib.Path(scores).parent.mkdir(parents=True, exist_ok=True)
    print(f"sampled {len(cells)} candidates across {len(tlist)} targets -> Boltz", flush=True)
    _run_batch(cells, out_dir, scores, samples, "gpu", True, batch_in)


if __name__ == "__main__":
    main()
