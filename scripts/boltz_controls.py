"""Boltz-2 discrimination control: can Boltz separate known binders from random
molecules for our targets? Calibrates whether the mismatch null is informative.

Each cell = a known or random ligand (reused from data/dock/dock_scores.csv) co-folded
into its target's OWN pocket sequence (from data/boltz/matrix_inputs.json). Reuses the
Boltz-invocation machinery of scripts.boltz_matrix. Idempotent: skips a (target, smiles)
cell already present in the scores CSV.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import shutil
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


def enumerate_control_cells(dock_scores_csv: str, inputs_json: str, cap: int | None = None) -> list[dict]:
    """Cells = each target's own-pocket known + random molecules co-folded into its own
    sequence. `cap` limits the number of molecules PER (target, class) to bound Boltz
    compute at scale (deterministic: first `cap` unique SMILES in CSV order)."""
    seq_of = {p["target_id"]: p["sequence"] for p in json.load(open(inputs_json))["proteins"]}
    df = pd.read_csv(dock_scores_csv)
    cells = []
    for tid, seq in seq_of.items():
        for cls in ("known", "random"):
            sub = df[(df.target == tid) & (df.pocket == tid) & (df.source == cls)]
            smis = list(sub.molecule.unique())
            if cap is not None:
                smis = smis[:cap]
            for smi in smis:
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


def parse_batch_result(batch_results_dir: str, stem: str) -> dict:
    """Parse one cell's outputs from a batch (directory-input) boltz run, where all records
    live under a single `boltz_results_<indir>/predictions/<stem>/` tree (vs the per-cell
    layout `boltz_results_<stem>/...`). Returns affinity_pred/binder_prob/ligand_iptm (NaN if absent)."""
    nan = float("nan")
    aff, prob, iptm = nan, nan, nan
    ap = glob.glob(os.path.join(batch_results_dir, "predictions", stem, f"affinity_{stem}*.json"))
    if ap:
        d = json.load(open(sorted(ap)[0]))
        aff = d.get("affinity_pred_value", nan)
        prob = d.get("affinity_probability_binary", nan)
    cp = glob.glob(os.path.join(batch_results_dir, "predictions", stem, f"confidence_{stem}*.json"))
    if cp:
        c = json.load(open(sorted(cp)[0]))
        iptm = c.get("ligand_iptm", c.get("iptm", nan))
    return {"affinity_pred": aff, "binder_prob": prob, "ligand_iptm": iptm}


def _run_batch(cells, out_dir, scores, samples, accelerator, no_kernels, batch_in):
    """Batch mode: write every not-yet-scored cell's YAML into one input dir and run
    `boltz predict <dir>` ONCE (single model load; MSA auto-deduped per unique sequence),
    then parse each cell's output and append. ~6x faster than one subprocess per cell."""
    pending = [c for c in cells if not cell_done(scores, c["target"], c["smiles"])]
    if not pending:
        print("batch: nothing pending — all cells already scored", flush=True)
        return
    indir = Path(batch_in)
    shutil.rmtree(indir, ignore_errors=True)
    indir.mkdir(parents=True)
    for c in pending:
        write_yaml(indir / f"{c['stem']}.yaml", c["sequence"], c["smiles"])
    print(f"batch: {len(pending)} pending cells -> one boltz run over {indir}/", flush=True)
    cmd = [BOLTZ, "predict", str(indir), "--use_msa_server", "--accelerator", accelerator,
           "--out_dir", out_dir, "--output_format", "pdb",
           "--diffusion_samples_affinity", str(samples)]
    if no_kernels:
        cmd.append("--no_kernels")
    subprocess.run(cmd, check=True)
    results_dir = os.path.join(out_dir, f"boltz_results_{indir.name}")
    n_ok = 0
    for c in pending:
        r = parse_batch_result(results_dir, c["stem"])
        if r["affinity_pred"] == r["affinity_pred"]:  # not NaN
            n_ok += 1
        _append_row(scores, {"target": c["target"], "smiles": c["smiles"], "class": c["class"], **r})
    print(f"batch: parsed {n_ok}/{len(pending)} cells with finite affinity -> {scores}", flush=True)


@click.command()
@click.option("--dock-scores", default="data/dock/dock_scores.csv")
@click.option("--inputs", default="data/boltz/matrix_inputs.json")
@click.option("--out-dir", default="boltz_out/controls")
@click.option("--scores", default="data/boltz/boltz_controls_scores.csv")
@click.option("--samples", default=3, type=int)
@click.option("--accelerator", default="mps")
@click.option("--limit", default=None, type=int, help="run only the first N cells (dry-run)")
@click.option("--cap", default=None, type=int, help="max molecules per (target,class) to bound compute")
@click.option("--no-kernels", "no_kernels", is_flag=True, default=False,
              help="pass --no_kernels to boltz (needed on CUDA boxes without a CUDA toolkit / "
                   "cuequivariance+triton kernels; standard PyTorch GPU path).")
@click.option("--batch", is_flag=True, default=False,
              help="batch mode: write all pending cells to one input dir and run boltz ONCE over it "
                   "(single model load instead of one subprocess/cell) — ~6x faster on GPU.")
@click.option("--batch-in", "batch_in", default="boltz_batch_in",
              help="input directory for batch mode (basename determines the boltz results dir).")
def main(dock_scores, inputs, out_dir, scores, samples, accelerator, limit, cap, no_kernels,
         batch, batch_in):
    cells = enumerate_control_cells(dock_scores, inputs, cap=cap)
    if limit is not None:
        cells = cells[:limit]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("boltz_in", exist_ok=True)
    os.makedirs(os.path.dirname(scores), exist_ok=True)

    if batch:
        _run_batch(cells, out_dir, scores, samples, accelerator, no_kernels, batch_in)
        return

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
            cmd = [BOLTZ, "predict", str(ypath), "--use_msa_server", "--accelerator", accelerator,
                   "--out_dir", out_dir, "--output_format", "pdb",
                   "--diffusion_samples_affinity", str(samples)]
            if no_kernels:
                cmd.append("--no_kernels")
            subprocess.run(cmd, check=True)
        r = parse_results(out_dir, stem)
        _append_row(scores, {"target": c["target"], "smiles": c["smiles"], "class": c["class"], **r})
        print(f"[{n}/{len(cells)}] {stem} aff={r['affinity_pred']} prob={r['binder_prob']} "
              f"iptm={r['ligand_iptm']}", flush=True)
    print(f"done: {scores}")


if __name__ == "__main__":
    main()
