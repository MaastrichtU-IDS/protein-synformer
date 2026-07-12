"""Generation-with-routes (SP-DPO pilot, Task 2).

Pocket-conditioned molecular generation that persists per-molecule route
tensors alongside SMILES so a later CPU docking step and a GPU DPO trainer
can hand off across venvs (see .superpowers/sdd/task-2-brief.md).

Runs in .venv-train (needs torch + the model stack). Writes:
  <out-prefix>.smi        one unique SMILES per line, in generation order.
  <out-prefix>.routes.pt  torch.save of {code, code_padding_mask, mols:[...]},
                           where mols[i] corresponds exactly to line i of the
                           .smi file (see dedup_keep_first / main below).

Mirrors the load sequence in scripts/dock_prepare.py's `generate-pocket`
command and scripts/generate_enriched.py's info-handling.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import click

# Repo root on sys.path so `scripts.*` / `synformer.*` import when run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def dedup_keep_first(records: list[dict]) -> list[dict]:
    """Dedup a list of dicts (each with a "smiles" key) by SMILES, keeping the
    first occurrence of each and preserving order. Pure — no torch/model deps."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        smi = r["smiles"]
        if smi in seen:
            continue
        seen.add(smi)
        out.append(r)
    return out


@click.command()
@click.option("--ckpt", required=True, help="Pocket-conditioned checkpoint to load.")
@click.option("--target", required=True, help="Target id (pocket .npz stem).")
@click.option("--pocket-dir", default="data/pockets", help="Directory of pocket .npz files.")
@click.option("--n", type=int, default=100, help="Target count of unique molecules.")
@click.option("--repeat", type=int, default=64, help="Batch size per sample_pocket() call.")
@click.option("--n-calls-max", type=int, default=None,
              help="Cap on sample_pocket() calls (default: generous multiple of n/repeat).")
@click.option("--seed", type=int, default=42)
@click.option("--out-prefix", required=True, help="Output path prefix (writes .smi + .routes.pt).")
def main(ckpt, target, pocket_dir, n, repeat, n_calls_max, seed, out_prefix):
    """Sample pocket-conditioned molecules for `target` and persist SMILES + per-molecule
    route tensors (token_types, rxn_indices, reactant_fps, token_padding_mask) plus the
    shared pocket code, so a CPU docking step and GPU DPO trainer can hand off across venvs."""
    import torch

    from scripts.sample_helpers import load_model, sample_pocket
    from synformer.data.pocket_io import load_pockets
    from synformer.molopt.dpo import routes_from_result

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fpindex, rxn_matrix = load_model(ckpt, None, device)
    pockets = load_pockets(pocket_dir)
    if target not in pockets:
        raise click.ClickException(f"target {target!r} not found in pocket-dir {pocket_dir}")

    n_calls_max = n_calls_max or max(4, (n // repeat) * 3)

    records: list[dict] = []
    pocket_code = None
    pocket_code_padding_mask = None
    calls = 0
    t0 = time.time()
    seen_smiles: set[str] = set()
    while len(seen_smiles) < n and calls < n_calls_max:
        calls += 1
        # sample_pocket returns (info, result): `info` is keyed by batch row index i,
        # already filtered to stack-depth-1 (valid) rows, built via
        # enumerate(result.build()) — so info[i] corresponds exactly to row i of
        # `result`. routes_from_result(result)[i] slices that same row i's route
        # tensors. This index alignment is the SMILES<->route correspondence; we
        # exploit it directly rather than re-deriving it.
        info, result = sample_pocket(
            target, model, fpindex, rxn_matrix, pockets, device, repeat=repeat,
        )
        routes = routes_from_result(result)
        if pocket_code is None:
            # The pocket code is identical across every row of a batch (one
            # pocket, repeated) — store it once, the DPO trainer tiles it.
            pocket_code = result.code[0:1].cpu()
            pocket_code_padding_mask = result.code_padding_mask[0:1].cpu()
        for i, d in info.items():
            route = routes[i]
            records.append({
                "smiles": d["smiles"],
                "token_types": route["token_types"],
                "rxn_indices": route["rxn_indices"],
                "reactant_fps": route["reactant_fps"],
                "token_padding_mask": route["token_padding_mask"],
            })
            seen_smiles.add(d["smiles"])
        click.echo(
            f"{target} call {calls}/{n_calls_max}: {len(records)} raw / "
            f"{len(seen_smiles)} unique so far ({time.time() - t0:.1f}s)",
        )

    mols = dedup_keep_first(records)[:n]

    out_smi = Path(f"{out_prefix}.smi")
    out_routes = Path(f"{out_prefix}.routes.pt")
    out_smi.parent.mkdir(parents=True, exist_ok=True)
    with open(out_smi, "w") as f:
        for m in mols:
            f.write(m["smiles"] + "\n")

    torch.save(
        {
            "code": pocket_code,
            "code_padding_mask": pocket_code_padding_mask,
            "mols": mols,
        },
        out_routes,
    )
    click.echo(
        f"WROTE {out_smi} ({len(mols)} SMILES) + {out_routes} "
        f"({calls} calls, {time.time() - t0:.1f}s total)",
    )


if __name__ == "__main__":
    main()
