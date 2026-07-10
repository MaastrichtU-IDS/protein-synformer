"""Generate pocket-conditioned candidates with optional enrichment weights.
Runs in .venv-train (GPU). Emits one JSON record per unique valid molecule:
{"smiles", "bb": [building-block fpindex ids], "tpl": [reaction-template ids]}."""
from __future__ import annotations

import json
import pathlib

import click

from synformer.molopt.enrich import EnrichWeights, molecule_index_sets


def load_weights(path: str | None) -> EnrichWeights | None:
    if path is None or path == "NONE":
        return None
    d = json.loads(pathlib.Path(path).read_text())
    return EnrichWeights(
        bb={int(k): float(v) for k, v in d.get("bb", {}).items()},
        tpl={int(k): float(v) for k, v in d.get("tpl", {}).items()},
    )


def stacks_to_records(stacks) -> list[dict]:
    out, seen = [], set()
    for st in stacks:
        if st.get_stack_depth() != 1:
            continue
        smi = st.get_one_top().smiles
        if not smi or smi in seen:
            continue
        seen.add(smi)
        bb, tpl = molecule_index_sets(st.get_mol_idx_seq(), st.get_rxn_idx_seq())
        out.append({"smiles": smi, "bb": sorted(bb), "tpl": sorted(tpl)})
    return out


@click.command()
@click.option("--ckpt", required=True)
@click.option("--target", required=True)
@click.option("--pocket-dir", default="data/pockets")
@click.option("--weights", default="NONE")
@click.option("--n", type=int, default=1000)
@click.option("--repeat", type=int, default=64)
@click.option("--seed", type=int, default=42)
@click.option("--out", required=True)
def main(ckpt, target, pocket_dir, weights, n, repeat, seed, out):
    import torch
    from scripts.sample_helpers import load_model, sample_pocket
    from synformer.data.pocket_io import load_pockets

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fpindex, rxn_matrix = load_model(ckpt, None, device)
    pockets = load_pockets(pocket_dir)
    ew = load_weights(weights)

    records, seen = [], set()
    calls = 0
    while len(records) < n and calls < max(4, (n // repeat) * 3):
        calls += 1
        # sample_pocket returns (info, result): `result` is the raw generation
        # result (needs .build() to get Stack objects); `info` is a dict keyed
        # by batch index, already filtered to stack_depth == 1 entries, each
        # holding the built Stack under "stack". Use that directly rather than
        # re-building from `result`.
        info, _result = sample_pocket(
            target, model, fpindex, rxn_matrix, pockets, device,
            repeat=repeat, enrich_weights=ew,
        )
        stacks = [v["stack"] for v in info.values()]
        for r in stacks_to_records(stacks):
            if r["smiles"] not in seen:
                seen.add(r["smiles"])
                records.append(r)
    with open(out, "w") as fh:
        for r in records[:n]:
            fh.write(json.dumps(r) + "\n")
    print(f"{target}: wrote {min(len(records), n)} records to {out} ({calls} calls)", flush=True)


if __name__ == "__main__":
    main()
