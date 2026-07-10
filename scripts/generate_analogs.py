"""Generate synthesizable analogs of seed molecules with SynFormer's analog sampler.
Runs in .venv-train (GPU). Emits one JSON record per unique analog:
{"smiles": analog, "seed": seed-it-came-from, "sim": similarity-to-seed}."""
from __future__ import annotations

import json
import pathlib

import click


def read_seeds(path: str | pathlib.Path) -> list[str]:
    return [ln.strip() for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]


def df_to_records(df) -> list[dict]:
    best: dict[str, dict] = {}
    for _, row in df.iterrows():
        smi = row["smiles"]
        sim = float(row["score"])
        if smi not in best or sim > best[smi]["sim"]:
            best[smi] = {"smiles": smi, "seed": str(row["target"]), "sim": sim}
    return list(best.values())


@click.command()
@click.option("--seeds", "seeds_path", required=True)
@click.option("--model", "model_path", required=True)
@click.option("--out", required=True)
@click.option("--search-width", type=int, default=24)
@click.option("--exhaustiveness", type=int, default=64)
@click.option("--time-limit", type=int, default=180)
@click.option("--num-gpus", type=int, default=1)
def main(seeds_path, model_path, out, search_width, exhaustiveness, time_limit, num_gpus):
    from synformer.chem.mol import Molecule
    from synformer.sampler.analog.parallel import run_parallel_sampling_return_smiles

    seeds = read_seeds(seeds_path)
    mols = [Molecule(s) for s in seeds]
    df = run_parallel_sampling_return_smiles(
        input=mols, model_path=pathlib.Path(model_path),
        search_width=search_width, exhaustiveness=exhaustiveness,
        num_gpus=num_gpus, num_workers_per_gpu=1, time_limit=time_limit,
    )
    records = df_to_records(df) if df is not None and len(df) else []
    with open(out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} analog records from {len(seeds)} seeds to {out}", flush=True)


if __name__ == "__main__":
    main()
