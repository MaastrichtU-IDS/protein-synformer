"""Boltz-2 affinity pilot: does a STRUCTURE-based co-folding scorer show the
target-specificity that the sequence-based DTI proxies lacked?

For a few short proteins, run Boltz-2 affinity on three conditions:
  gen_correct    - generated molecule vs its correct protein
  gen_mismatch   - SAME generated molecule vs a DIFFERENT (mismatched) protein
  known_correct  - a known ligand vs the correct protein

Decisive test: gen_correct vs gen_mismatch. If Boltz-2 affinity differs meaningfully
(unlike DeepPurpose, which was identical), it is target-sensitive. Idempotent: skips a
condition whose affinity JSON already exists (so the earlier smoke is reused).

CPU-only on this Mac (~55 min/run); intended as an overnight background job.
"""
import json
import subprocess
from pathlib import Path

import click
import pandas as pd
import yaml

BOLTZ = ".venv-boltz/bin/boltz"


def write_yaml(path, seq, smiles):
    doc = {"version": 1,
           "sequences": [{"protein": {"id": "A", "sequence": seq}},
                         {"ligand": {"id": "L", "smiles": smiles}}],
           "properties": [{"affinity": {"binder": "L"}}]}
    yaml.safe_dump(doc, open(path, "w"), sort_keys=False)


def affinity_json_path(out_dir, stem):
    return Path(out_dir) / f"boltz_results_{stem}" / "predictions" / stem / f"affinity_{stem}.json"


def run_condition(stem, seq, smiles, out_dir, samples):
    ypath = Path("boltz_in") / f"{stem}.yaml"
    write_yaml(ypath, seq, smiles)
    ajson = affinity_json_path(out_dir, stem)
    if not ajson.exists():
        subprocess.run(
            [BOLTZ, "predict", str(ypath), "--use_msa_server", "--accelerator", "cpu",
             "--out_dir", out_dir, "--output_format", "pdb",
             "--diffusion_samples_affinity", str(samples)],
            check=True,
        )
    d = json.load(open(ajson))
    return d["affinity_pred_value"], d["affinity_probability_binary"]


@click.command()
@click.option("--inputs", default="data/evaluations/boltz_pilot_inputs.json")
@click.option("--n", default=3, type=int, help="number of proteins")
@click.option("--out-dir", default="boltz_out/pilot")
@click.option("--samples", default=3, type=int, help="affinity diffusion samples")
@click.option("--out", default="data/evaluations/boltz_pilot_results.csv")
def main(inputs, n, out_dir, samples, out):
    rows = json.load(open(inputs))[:n]
    seq_of = {r["target"]: r["seq"] for r in json.load(open(inputs))}
    Path("boltz_in").mkdir(exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for r in rows:
        t = r["target"]
        conditions = {
            "gen_correct": (r["seq"], r["gen_smiles"]),
            "gen_mismatch": (seq_of[r["mismatch_target"]], r["gen_smiles"]),
            "known_correct": (r["seq"], r["known_smiles"]),
        }
        for cond, (seq, smiles) in conditions.items():
            stem = f"{t}_{cond}"
            print(f"[{stem}] running (seq_len={len(seq)}) ...", flush=True)
            val, prob = run_condition(stem, seq, smiles, out_dir, samples)
            results.append({"target": t, "condition": cond,
                            "aff_pred_value_log10IC50uM": val, "aff_prob_binary": prob})
            print(f"[{stem}] pred_value={val:.3f} prob_binary={prob:.3f}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(out, index=False)
    print("\n=== Boltz-2 pilot ===")
    print(df.pivot(index="target", columns="condition", values="aff_pred_value_log10IC50uM"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
