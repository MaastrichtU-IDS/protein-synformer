"""Run the 5x5 Boltz-2 co-folding mismatch grid on MPS.

Cell (hit target i, protein j): co-fold target i's top-1 docking hit with protein j's
sequence; record affinity_pred_value, affinity_probability_binary, ligand_iptm.
Idempotent: skips a (hit_target, protein) cell already present in the scores CSV.
The full 25-cell run is intended for the user's terminal (caffeinate); see the plan.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import subprocess
from pathlib import Path

import click
import pandas as pd
import yaml

BOLTZ = ".venv-boltz-mps/bin/boltz"
COLUMNS = ["hit_target", "protein", "smiles", "affinity_pred", "binder_prob", "ligand_iptm"]


def enumerate_cells(inputs: dict) -> list[dict]:
    seq_of = {p["target_id"]: p["sequence"] for p in inputs["proteins"]}
    cells = []
    for hit in inputs["hits"]:
        i = hit["target_id"]
        for j, seq in seq_of.items():
            cells.append({"hit_target": i, "protein": j, "smiles": hit["smiles"],
                          "sequence": seq, "stem": f"{i}_into_{j}"})
    return cells


def write_yaml(path: Path, seq: str, smiles: str) -> None:
    doc = {"version": 1,
           "sequences": [{"protein": {"id": "A", "sequence": seq}},
                         {"ligand": {"id": "L", "smiles": smiles}}],
           "properties": [{"affinity": {"binder": "L"}}]}
    yaml.safe_dump(doc, open(path, "w"), sort_keys=False)


def _first_json(out_dir: str, stem: str, kind: str):
    # kind: "affinity" or "confidence"
    pat = os.path.join(out_dir, f"boltz_results_{stem}", "predictions", stem, f"{kind}_{stem}*.json")
    hits = sorted(glob.glob(pat))
    return hits[0] if hits else None


def parse_results(out_dir: str, stem: str) -> dict:
    nan = float("nan")
    aff, prob, iptm = nan, nan, nan
    ajson = _first_json(out_dir, stem, "affinity")
    if ajson:
        d = json.load(open(ajson))
        aff = d.get("affinity_pred_value", nan)
        prob = d.get("affinity_probability_binary", nan)
    cjson = _first_json(out_dir, stem, "confidence")
    if cjson:
        c = json.load(open(cjson))
        iptm = c.get("ligand_iptm", c.get("iptm", nan))
    return {"affinity_pred": aff, "binder_prob": prob, "ligand_iptm": iptm}


def cell_done(scores_csv: str, hit_target: str, protein: str) -> bool:
    if not os.path.exists(scores_csv):
        return False
    df = pd.read_csv(scores_csv)
    if df.empty:
        return False
    return not df[(df.hit_target == hit_target) & (df.protein == protein)].empty


def _append_row(scores_csv: str, row: dict) -> None:
    new = not os.path.exists(scores_csv) or os.path.getsize(scores_csv) == 0
    with open(scores_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in COLUMNS})
        fh.flush()


@click.command()
@click.option("--inputs", default="data/boltz/matrix_inputs.json")
@click.option("--out-dir", default="boltz_out/matrix")
@click.option("--scores", default="data/boltz/boltz_scores.csv")
@click.option("--samples", default=3, type=int, help="diffusion_samples_affinity")
@click.option("--accelerator", default="mps")
@click.option("--limit", default=None, type=int, help="run only the first N cells (dry-run)")
def main(inputs, out_dir, scores, samples, accelerator, limit):
    data = json.load(open(inputs))
    cells = enumerate_cells(data)
    if limit is not None:
        cells = cells[:limit]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("boltz_in", exist_ok=True)
    os.makedirs(os.path.dirname(scores), exist_ok=True)

    for n, c in enumerate(cells, 1):
        stem = c["stem"]
        if cell_done(scores, c["hit_target"], c["protein"]):
            print(f"[{n}/{len(cells)}] skip {stem} (already scored)", flush=True)
            continue
        ypath = Path("boltz_in") / f"{stem}.yaml"
        write_yaml(ypath, c["sequence"], c["smiles"])
        if not _first_json(out_dir, stem, "affinity"):
            print(f"[{n}/{len(cells)}] running {stem} (seq_len={len(c['sequence'])}) ...", flush=True)
            subprocess.run(
                [BOLTZ, "predict", str(ypath), "--use_msa_server", "--accelerator", accelerator,
                 "--out_dir", out_dir, "--output_format", "pdb",
                 "--diffusion_samples_affinity", str(samples)],
                check=True,
            )
        r = parse_results(out_dir, stem)
        _append_row(scores, {"hit_target": c["hit_target"], "protein": c["protein"],
                             "smiles": c["smiles"], **r})
        print(f"[{n}/{len(cells)}] {stem} aff={r['affinity_pred']} prob={r['binder_prob']} "
              f"iptm={r['ligand_iptm']}", flush=True)
    print(f"done: {scores}")


if __name__ == "__main__":
    main()
