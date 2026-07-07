"""Boltz-2 discrimination control: can Boltz separate known binders from random
molecules for our targets? Calibrates whether the mismatch null is informative.

Each cell = a known or random ligand (reused from data/dock/dock_scores.csv) co-folded
into its target's OWN pocket sequence (from data/boltz/matrix_inputs.json). Reuses the
Boltz-invocation machinery of scripts.boltz_matrix. Idempotent: skips a (target, smiles)
cell already present in the scores CSV.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import click
import pandas as pd

from scripts.boltz_matrix import BOLTZ, parse_results, write_yaml

COLUMNS = ["target", "smiles", "class", "affinity_pred", "binder_prob", "ligand_iptm"]


def stem_for(target: str, cls: str, smiles: str) -> str:
    """Stable, target-scoped stem for a control cell (same SMILES + different target =
    different co-fold = different stem)."""
    h = hashlib.md5(smiles.encode()).hexdigest()[:8]
    return f"ctrl_{target}_{cls}_{h}"


def enumerate_control_cells(dock_scores_csv: str, inputs_json: str) -> list[dict]:
    seq_of = {p["target_id"]: p["sequence"] for p in json.load(open(inputs_json))["proteins"]}
    df = pd.read_csv(dock_scores_csv)
    cells = []
    for tid, seq in seq_of.items():
        for cls in ("known", "random"):
            sub = df[(df.target == tid) & (df.pocket == tid) & (df.source == cls)]
            for smi in sub.molecule.unique():
                cells.append({"target": tid, "class": cls, "smiles": smi,
                              "sequence": seq, "stem": stem_for(tid, cls, smi)})
    return cells


def cell_done(scores_csv: str, target: str, smiles: str) -> bool:
    if not os.path.exists(scores_csv):
        return False
    df = pd.read_csv(scores_csv)
    if df.empty:
        return False
    return not df[(df.target == target) & (df.smiles == smiles)].empty


def _append_row(scores_csv: str, row: dict) -> None:
    new = not os.path.exists(scores_csv) or os.path.getsize(scores_csv) == 0
    with open(scores_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in COLUMNS})
        fh.flush()


@click.command()
@click.option("--dock-scores", default="data/dock/dock_scores.csv")
@click.option("--inputs", default="data/boltz/matrix_inputs.json")
@click.option("--out-dir", default="boltz_out/controls")
@click.option("--scores", default="data/boltz/boltz_controls_scores.csv")
@click.option("--samples", default=3, type=int)
@click.option("--accelerator", default="mps")
@click.option("--limit", default=None, type=int, help="run only the first N cells (dry-run)")
def main(dock_scores, inputs, out_dir, scores, samples, accelerator, limit):
    cells = enumerate_control_cells(dock_scores, inputs)
    if limit is not None:
        cells = cells[:limit]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("boltz_in", exist_ok=True)
    os.makedirs(os.path.dirname(scores), exist_ok=True)

    for n, c in enumerate(cells, 1):
        stem = c["stem"]
        if cell_done(scores, c["target"], c["smiles"]):
            print(f"[{n}/{len(cells)}] skip {stem} (already scored)", flush=True)
            continue
        ypath = Path("boltz_in") / f"{stem}.yaml"
        write_yaml(ypath, c["sequence"], c["smiles"])
        from scripts.boltz_matrix import _first_json
        if not _first_json(out_dir, stem, "affinity"):
            print(f"[{n}/{len(cells)}] running {stem} ({c['class']}) ...", flush=True)
            subprocess.run(
                [BOLTZ, "predict", str(ypath), "--use_msa_server", "--accelerator", accelerator,
                 "--out_dir", out_dir, "--output_format", "pdb",
                 "--diffusion_samples_affinity", str(samples)],
                check=True,
            )
        r = parse_results(out_dir, stem)
        _append_row(scores, {"target": c["target"], "smiles": c["smiles"], "class": c["class"], **r})
        print(f"[{n}/{len(cells)}] {stem} aff={r['affinity_pred']} prob={r['binder_prob']} "
              f"iptm={r['ligand_iptm']}", flush=True)
    print(f"done: {scores}")


if __name__ == "__main__":
    main()
