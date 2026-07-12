"""DPO training loop over preference pairs (SP-DPO pilot, Task 4 Step 1).

Loads a pocket-conditioned SP-C checkpoint twice: once as the trainable
`policy`, once as a frozen `reference` (`requires_grad_(False)`, called only
under `torch.no_grad()`). For every target present in BOTH `--routes-dir`
(`<target>.routes.pt`, from `scripts/generate_routes.py`) and `--pairs-dir`
(`pairs_<target>.json`, from `scripts/dpo_pairs.py`), looks up each
preference pair's winner/loser SMILES in that target's route file, computes
per-molecule total log-likelihoods under policy (with grad) and reference
(no grad) via `model.get_log_likelihood(...)["total"].sum(dim=1)`, and
back-propagates `synformer.molopt.dpo.dpo_loss` into the policy only.

Batching choice: ONE (winner, loser) PAIR AT A TIME per
`get_log_likelihood()` call (batch dim = 1), because routes for different
molecules have different token-sequence lengths and this pilot does not
build a padding collator. Multiple pairs still share one optimizer step:
`dpo_train_step` loops over every pair in its `batch` argument, accumulates
gradients (mean loss over the pairs), and calls `optimizer.step()` once at
the end — so a "step" here means one gradient update per target-batch of
pairs, not one update per pair.

`dpo_train_step` (and the `pair_log_likelihoods` / `build_pair_batch_item` /
`subsample_pairs` helpers it's built from) is AGNOSTIC to whether
`policy`/`reference` are the real `Synformer` or any stand-in exposing the
same `get_log_likelihood(...)` call signature — see `tests/test_dpo_train.py`
for the CPU stand-in this is TDD'd against. `main()` (the only place that
touches the real model / heavy deps) is not unit-tested here; it is exercised
by the controller's ops run.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click
import torch

# Repo root on sys.path so `scripts.*` / `synformer.*` import when run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synformer.molopt.dpo import dpo_loss  # noqa: E402


def pair_log_likelihoods(model, code, code_padding_mask, route: dict) -> torch.Tensor:
    """Per-molecule total log-likelihood for one route under `model`.

    `route` has the four decoder-input tensors persisted by
    `generate_routes.py` (`token_types`, `rxn_indices`, `reactant_fps`,
    `token_padding_mask`), each shaped `[1, seq]` (batch dim = 1 — see module
    docstring). Returns a 1-D tensor of shape `[1]`: the per-token
    log-likelihoods (`result["total"]`, shape `[1, seq-1]`) summed over the
    sequence, exactly as `scripts/sample_helpers.py::sample_pocket` does per
    row (`ll["total"][i].sum()`).
    """
    result = model.get_log_likelihood(
        code=code,
        code_padding_mask=code_padding_mask,
        token_types=route["token_types"],
        rxn_indices=route["rxn_indices"],
        reactant_fps=route["reactant_fps"],
        token_padding_mask=route["token_padding_mask"],
    )
    return result["total"].sum(dim=1)


def dpo_train_step(policy, reference, batch: list[dict], optimizer, beta: float = 0.1) -> dict:
    """One optimizer step of DPO over `batch`.

    `batch` is a list of pair-dicts, each `{"code", "code_padding_mask",
    "winner", "loser"}` (see `build_pair_batch_item`). For every pair:
    policy log-likelihoods are computed WITH grad, reference log-likelihoods
    under `torch.no_grad()` (reference is frozen — its parameters must
    already have `requires_grad=False`, and this function never touches
    them). `dpo_loss` is backpropagated per pair (mean-reduced over the
    batch) before a single `optimizer.step()` — see module docstring for why
    this is one gradient update, not `len(batch)` of them.

    Returns `{"loss", "margin", "drift"}` over `batch`, all evaluated from the
    forward passes taken BEFORE `optimizer.step()` is applied (i.e. pre-step
    values, useful for per-epoch logging):

    - `loss`   — mean DPO loss.
    - `margin` — mean implicit-reward margin
      `mean((llpi_w-llref_w)-(llpi_l-llref_l))`; a positive, growing margin =
      policy learning to prefer winners over losers.
    - `drift`  — mean policy-minus-reference log-prob SHIFT over both winners
      AND losers, `mean(((llpi_w-llref_w)+(llpi_l-llref_l))/2)`. This is the
      KL-to-reference proxy / reward-collapse monitor the brief requires:
      unlike `margin` (a DIFFERENCE that can grow while the policy drifts far
      from the reference), `drift` is the SUM/average shift, so a large-
      magnitude `drift` — especially a strongly negative one — flags the
      policy moving away from the reference (assigning both winners and losers
      much lower probability than the reference), the classic DPO reward-
      collapse failure mode. Watch it stay near 0 during the ops run.
    """
    optimizer.zero_grad()
    n = len(batch)
    losses = []
    margins = []
    drifts = []
    for pair in batch:
        code = pair["code"]
        code_padding_mask = pair["code_padding_mask"]
        llpi_w = pair_log_likelihoods(policy, code, code_padding_mask, pair["winner"])
        llpi_l = pair_log_likelihoods(policy, code, code_padding_mask, pair["loser"])
        with torch.no_grad():
            llref_w = pair_log_likelihoods(reference, code, code_padding_mask, pair["winner"])
            llref_l = pair_log_likelihoods(reference, code, code_padding_mask, pair["loser"])
        loss = dpo_loss(llpi_w, llpi_l, llref_w, llref_l, beta=beta)
        (loss / n).backward()
        losses.append(loss.item())
        shift_w = (llpi_w - llref_w).item()
        shift_l = (llpi_l - llref_l).item()
        margins.append(shift_w - shift_l)
        drifts.append((shift_w + shift_l) / 2)
    optimizer.step()
    return {"loss": sum(losses) / n, "margin": sum(margins) / n, "drift": sum(drifts) / n}


def build_out_checkpoint(base_hparams, policy_state_dict) -> dict:
    """Wrap a trained policy's `state_dict` into a `load_model`-compatible
    checkpoint blob.

    `load_model(ckpt, config_path=None, ...)` reads `ckpt["hyper_parameters"]
    ["config"]` and `ckpt["state_dict"]` with keys prefixed `"model."` (which
    it strips via `k[6:]`). Saving a bare `policy.state_dict()` would produce
    a checkpoint NONE of the downstream consumers (generate_routes.py,
    sample_pocket, this script's own `--ckpt` load) could reload. This helper
    re-attaches the `"model."` prefix to every parameter key and carries the
    original `hyper_parameters` through unchanged, so the DPO'd checkpoint is
    a drop-in replacement for the base checkpoint.
    """
    return {
        "hyper_parameters": base_hparams,
        "state_dict": {f"model.{k}": v for k, v in policy_state_dict.items()},
    }


def build_pair_batch_item(routes_by_smiles: dict, code, code_padding_mask, winner_smiles: str, loser_smiles: str):
    """Build one `dpo_train_step` batch entry for a (winner, loser) SMILES
    pair, looking up each SMILES's route in `routes_by_smiles`
    (`{smiles: route_dict}`, built from a target's `.routes.pt` `mols` list).

    Returns `None` if either SMILES is missing from the target's routes file
    — callers should count and log these rather than crash: `pairs_<target>
    .json` (Task 3) is built from a docked subset of a target's generated
    molecules and is not guaranteed to be a subset of a *particular*
    `generate_routes.py` run's (deduped, possibly truncated) output.
    """
    winner = routes_by_smiles.get(winner_smiles)
    loser = routes_by_smiles.get(loser_smiles)
    if winner is None or loser is None:
        return None
    return {"code": code, "code_padding_mask": code_padding_mask, "winner": winner, "loser": loser}


def subsample_pairs(pairs: list, max_pairs: int, seed: int) -> tuple[list, int]:
    """Deterministically cap `pairs` to at most `max_pairs` elements.

    `make_pairs` (Task 3) can emit ~k^2 pairs per target (up to ~900 for a
    target with a large docked pool), which is a lot of per-pair decoder
    forward passes for a pilot run. Subsampling uses a seeded `random.Random`
    so the same `(pairs, max_pairs, seed)` always yields the same subset
    (reproducible runs), and returns `kept` in the ORIGINAL relative order
    (stable diffs / no shuffling surprise in logs). Returns
    `(kept, n_dropped)` — callers must log `n_dropped` rather than silently
    truncate.
    """
    if len(pairs) <= max_pairs:
        return list(pairs), 0
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    keep_idx = sorted(indices[:max_pairs])
    kept = [pairs[i] for i in keep_idx]
    return kept, len(pairs) - max_pairs


@click.command()
@click.option("--ckpt", required=True, help="Pocket-conditioned SP-C checkpoint to fine-tune.")
@click.option("--routes-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Directory of <target>.routes.pt files (scripts/generate_routes.py).")
@click.option("--pairs-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Directory of pairs_<target>.json files (scripts/dpo_pairs.py).")
@click.option("--out-ckpt", required=True, help="Where to save the DPO'd policy state_dict.")
@click.option("--lr", type=float, default=1e-5)
@click.option("--epochs", type=int, default=1)
@click.option("--beta", type=float, default=0.1)
@click.option("--max-pairs-per-target", type=int, default=200,
              help="Deterministic cap on pairs used per target per epoch (see subsample_pairs).")
@click.option("--seed", type=int, default=42)
def main(ckpt, routes_dir, pairs_dir, out_ckpt, lr, epochs, beta, max_pairs_per_target, seed):
    """DPO fine-tune `--ckpt` on the preference pairs in `--pairs-dir`, using
    the route tensors in `--routes-dir`; save the fine-tuned state_dict to
    `--out-ckpt`. See module docstring for the batching/gradient-accumulation
    design."""
    from scripts.sample_helpers import load_model

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read the raw checkpoint once (in addition to the two load_model() calls
    # below) so `--out-ckpt` can be saved in the SAME Lightning-style shape
    # (`{"hyper_parameters", "state_dict"}` with a "model." key prefix) that
    # `load_model(..., config_path=None, ...)` expects on read-back. Saving a
    # bare `state_dict()` would make the DPO'd checkpoint unloadable by every
    # downstream consumer (generate_routes.py, sample_pocket, this script
    # itself) — see load_model's `config_path is None` branch.
    base_ckpt = torch.load(ckpt, map_location="cpu")
    if not all(k.startswith("model.") for k in base_ckpt["state_dict"]):
        raise click.ClickException(
            f"{ckpt}: state_dict keys don't all start with 'model.' — load_model's 6-char "
            "prefix-strip assumption doesn't hold for this checkpoint; refusing to write a "
            "DPO'd checkpoint the pipeline couldn't reload.",
        )

    policy, _, _ = load_model(ckpt, None, device)
    policy.train()

    # Frozen reference: a second, independent load of the same checkpoint
    # (simplest way to guarantee no parameter aliasing with the policy).
    reference, _, _ = load_model(ckpt, None, device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    routes_dir = Path(routes_dir)
    pairs_dir = Path(pairs_dir)

    route_files = {f.name[: -len(".routes.pt")]: f for f in routes_dir.glob("*.routes.pt")}
    pair_files = {f.name[len("pairs_"): -len(".json")]: f for f in pairs_dir.glob("pairs_*.json")}
    targets = sorted(set(route_files) & set(pair_files))
    if not targets:
        raise click.ClickException(
            f"no target present in both --routes-dir ({routes_dir}) and --pairs-dir ({pairs_dir})",
        )
    click.echo(f"targets: {targets}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_losses = []
        epoch_margins = []
        epoch_drifts = []
        for target in targets:
            routes_blob = torch.load(route_files[target], map_location=device)
            code = routes_blob["code"]
            code_padding_mask = routes_blob["code_padding_mask"]
            routes_by_smiles = {m["smiles"]: m for m in routes_blob["mols"]}

            raw_pairs = json.loads(pair_files[target].read_text())
            capped, n_dropped = subsample_pairs(raw_pairs, max_pairs_per_target, seed)
            if n_dropped:
                click.echo(
                    f"{target}: capped to {len(capped)}/{len(raw_pairs)} pairs "
                    f"(dropped {n_dropped}, seed={seed}, --max-pairs-per-target={max_pairs_per_target})",
                )

            batch_items = []
            n_missing = 0
            for winner_smiles, loser_smiles in capped:
                item = build_pair_batch_item(routes_by_smiles, code, code_padding_mask, winner_smiles, loser_smiles)
                if item is None:
                    n_missing += 1
                    continue
                batch_items.append(item)
            if n_missing:
                click.echo(f"{target}: skipped {n_missing}/{len(capped)} pairs (winner/loser SMILES missing from routes file)")

            if not batch_items:
                click.echo(f"{target}: 0 usable pairs, skipping")
                continue

            stats = dpo_train_step(policy, reference, batch_items, optimizer, beta=beta)
            epoch_losses.append(stats["loss"])
            epoch_margins.append(stats["margin"])
            epoch_drifts.append(stats["drift"])
            click.echo(
                f"epoch {epoch} {target}: n_pairs={len(batch_items)} "
                f"loss={stats['loss']:.4f} margin={stats['margin']:.4f} drift={stats['drift']:.4f}",
            )

        if epoch_losses:
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            mean_margin = sum(epoch_margins) / len(epoch_margins)
            mean_drift = sum(epoch_drifts) / len(epoch_drifts)
            click.echo(
                f"epoch {epoch} MEAN over {len(epoch_losses)} targets: "
                f"loss={mean_loss:.4f} margin={mean_margin:.4f} drift={mean_drift:.4f} "
                f"(drift = policy-vs-reference log-prob shift; large |drift| flags reward collapse)",
            )
        else:
            click.echo(f"epoch {epoch}: no target had usable pairs")

    out_path = Path(out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_out_checkpoint(base_ckpt["hyper_parameters"], policy.state_dict()), out_path)
    click.echo(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
