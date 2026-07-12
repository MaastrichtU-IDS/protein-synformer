"""TDD for scripts/dpo_train.py's train step.

Uses a tiny CPU stand-in model (NOT the real Synformer) that exposes the same
`get_log_likelihood(...)` call signature the real model uses, so the tests
stay cheap and exercise exactly the training-step logic in isolation from any
GPU/model-loading concerns (those are exercised only by the later ops run).

Run with: .venv/bin/python -m pytest tests/test_dpo_train.py -q
"""
from __future__ import annotations

import torch
import torch.nn as nn

from scripts.dpo_train import (
    build_out_checkpoint,
    build_pair_batch_item,
    dpo_train_step,
    pair_log_likelihoods,
    subsample_pairs,
)


class _StandInModel(nn.Module):
    """Minimal stand-in for Synformer's `get_log_likelihood`.

    Ignores route content except a scalar "marker" tag stashed at
    `token_types[0, 0]` (+1.0 for winner-tagged routes, -1.0 for loser-tagged
    routes — see `_route` below). Returns
        total = theta * marker   (broadcast over `seq_len - 1` token columns)
    so `total.sum(dim=1)` grows with theta for winner routes (marker=+1) and
    shrinks with theta for loser routes (marker=-1). Because
        d/dtheta [ llpi_w - llpi_l ] = (seq_len-1) * (marker_w - marker_l)
                                     = (seq_len-1) * 2 > 0,
    any Adam step that increases theta (which is exactly what minimizing the
    DPO loss does, since the DPO loss decreases as (llpi_w - llpi_l) grows)
    provably raises the winner ll and lowers the loser ll on the very same
    fixed batch — the property the tests below assert.
    """

    def __init__(self, theta: float = 0.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(theta)))

    def get_log_likelihood(self, code, code_padding_mask, token_types, rxn_indices, reactant_fps, token_padding_mask):
        marker = token_types[0, 0].float()
        seq_len = token_types.shape[1]
        total = self.theta * marker * torch.ones(token_types.shape[0], seq_len - 1)
        return {"total": total}


def _route(marker: float, seq_len: int = 4) -> dict:
    token_types = torch.zeros(1, seq_len)
    token_types[0, 0] = marker
    return {
        "token_types": token_types,
        "rxn_indices": torch.zeros(1, seq_len),
        "reactant_fps": torch.zeros(1, seq_len, 1),
        "token_padding_mask": torch.zeros(1, seq_len, dtype=torch.bool),
    }


def _make_batch(n_pairs: int = 2) -> list[dict]:
    code = torch.zeros(1, 1, 1)
    code_padding_mask = torch.zeros(1, 1, dtype=torch.bool)
    return [
        {"code": code, "code_padding_mask": code_padding_mask, "winner": _route(1.0), "loser": _route(-1.0)}
        for _ in range(n_pairs)
    ]


def _batch_stats(policy, reference, batch, beta):
    """Recompute mean loss + mean implicit-reward margin for `batch` WITHOUT
    taking a gradient step — used to compare before/after `dpo_train_step`."""
    from synformer.molopt.dpo import dpo_loss

    losses, margins = [], []
    for pair in batch:
        llpi_w = pair_log_likelihoods(policy, pair["code"], pair["code_padding_mask"], pair["winner"])
        llpi_l = pair_log_likelihoods(policy, pair["code"], pair["code_padding_mask"], pair["loser"])
        with torch.no_grad():
            llref_w = pair_log_likelihoods(reference, pair["code"], pair["code_padding_mask"], pair["winner"])
            llref_l = pair_log_likelihoods(reference, pair["code"], pair["code_padding_mask"], pair["loser"])
        losses.append(dpo_loss(llpi_w, llpi_l, llref_w, llref_l, beta=beta).item())
        margins.append(((llpi_w - llref_w) - (llpi_l - llref_l)).item())
    return sum(losses) / len(losses), sum(margins) / len(margins)


def _approx(x, abs_tol=1e-5):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < abs_tol

    return _Approx()


def test_dpo_train_step_decreases_loss_and_increases_margin_and_freezes_reference():
    policy = _StandInModel(theta=0.0)
    reference = _StandInModel(theta=0.0)
    for p in reference.parameters():
        p.requires_grad_(False)

    batch = _make_batch(n_pairs=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)

    loss_before, margin_before = _batch_stats(policy, reference, batch, beta=0.1)
    ref_theta_before = reference.theta.detach().clone()

    stats = dpo_train_step(policy, reference, batch, optimizer, beta=0.1)

    # 1. DPO loss on the same fixed batch decreases after one Adam step.
    loss_after, margin_after = _batch_stats(policy, reference, batch, beta=0.1)
    assert loss_after < loss_before
    # dpo_train_step's own returned loss (pre-step forward pass) matches the
    # independently recomputed pre-step loss.
    assert stats["loss"] == _approx(loss_before)

    # 2. Reference stand-in is unchanged and frozen.
    assert torch.equal(reference.theta, ref_theta_before)
    assert not reference.theta.requires_grad

    # 3. Implicit-reward margin increases after the step.
    assert margin_after > margin_before

    # 4. The returned dict carries the drift (KL-to-reference / collapse) proxy.
    assert "drift" in stats


def test_dpo_train_step_reports_drift_that_moves():
    """The symmetric batch above keeps drift == 0 by construction (winner up
    exactly as much as loser down). Use an ASYMMETRIC batch (winner marker
    +2.0, loser marker -1.0) so the mean policy-vs-reference shift is genuinely
    nonzero, and assert `drift` moves off its initial value after a step."""
    policy = _StandInModel(theta=0.0)
    reference = _StandInModel(theta=0.0)
    for p in reference.parameters():
        p.requires_grad_(False)

    code = torch.zeros(1, 1, 1)
    code_padding_mask = torch.zeros(1, 1, dtype=torch.bool)
    batch = [{"code": code, "code_padding_mask": code_padding_mask, "winner": _route(2.0), "loser": _route(-1.0)}]
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)

    # policy == reference initially -> pre-step drift is exactly 0.
    stats0 = dpo_train_step(policy, reference, batch, optimizer, beta=0.1)
    assert stats0["drift"] == _approx(0.0)

    # After the first step theta > 0, so winners (+2) and losers (-1) no longer
    # cancel -> the pre-step drift of the SECOND call is nonzero (moved).
    stats1 = dpo_train_step(policy, reference, batch, optimizer, beta=0.1)
    assert stats1["drift"] != stats0["drift"]
    assert abs(stats1["drift"]) > 1e-6


def test_subsample_pairs_no_op_when_under_cap():
    pairs = [("w0", "l0"), ("w1", "l1")]
    kept, n_dropped = subsample_pairs(pairs, max_pairs=200, seed=42)
    assert kept == pairs
    assert n_dropped == 0


def test_subsample_pairs_caps_deterministically_and_reports_dropped():
    pairs = [(f"w{i}", f"l{i}") for i in range(30)]
    kept1, n_dropped1 = subsample_pairs(pairs, max_pairs=10, seed=42)
    kept2, n_dropped2 = subsample_pairs(pairs, max_pairs=10, seed=42)

    assert len(kept1) == 10
    assert n_dropped1 == 20
    # deterministic: same seed -> identical subsample
    assert kept1 == kept2
    assert n_dropped1 == n_dropped2
    # every kept pair came from the original list (no fabrication)
    assert all(p in pairs for p in kept1)


def test_subsample_pairs_different_seed_can_differ():
    pairs = [(f"w{i}", f"l{i}") for i in range(30)]
    kept_a, _ = subsample_pairs(pairs, max_pairs=10, seed=1)
    kept_b, _ = subsample_pairs(pairs, max_pairs=10, seed=2)
    assert kept_a != kept_b  # different seeds sample different subsets (overwhelmingly likely for 30 choose 10)


def test_build_pair_batch_item_found():
    code = torch.zeros(1, 1, 1)
    code_padding_mask = torch.zeros(1, 1, dtype=torch.bool)
    routes_by_smiles = {"CCO": _route(1.0), "CCN": _route(-1.0)}
    item = build_pair_batch_item(routes_by_smiles, code, code_padding_mask, "CCO", "CCN")
    assert item is not None
    assert item["winner"] is routes_by_smiles["CCO"]
    assert item["loser"] is routes_by_smiles["CCN"]
    assert item["code"] is code


def test_build_pair_batch_item_missing_winner_or_loser_returns_none():
    routes_by_smiles = {"CCO": _route(1.0)}
    code = torch.zeros(1, 1, 1)
    code_padding_mask = torch.zeros(1, 1, dtype=torch.bool)
    assert build_pair_batch_item(routes_by_smiles, code, code_padding_mask, "CCO", "NOPE") is None
    assert build_pair_batch_item(routes_by_smiles, code, code_padding_mask, "NOPE", "CCO") is None
    assert build_pair_batch_item(routes_by_smiles, code, code_padding_mask, "NOPE", "ALSO_NOPE") is None


def test_build_out_checkpoint_is_load_model_compatible():
    # load_model(ckpt, config_path=None, ...) reads ckpt["hyper_parameters"]["config"] and
    # ckpt["state_dict"] with keys prefixed "model." (stripped via k[6:]). This guards the
    # bug class caught once already: saving a bare state_dict would be unloadable downstream.
    base_hparams = {"config": {"model": {"dim": 8}, "chem": {"fpindex": "x", "rxn_matrix": "y"}}}
    policy_state_dict = {"encoder.weight": torch.zeros(2, 2), "head.bias": torch.ones(3)}

    blob = build_out_checkpoint(base_hparams, policy_state_dict)

    # hyper_parameters carried through unchanged (identity, not a copy — cheap and correct).
    assert blob["hyper_parameters"] is base_hparams
    # every state_dict key is "model."-prefixed and strips back to the original param name.
    assert set(blob["state_dict"].keys()) == {"model.encoder.weight", "model.head.bias"}
    for k in blob["state_dict"]:
        assert k.startswith("model.")
    stripped = {k[6:]: v for k, v in blob["state_dict"].items()}
    assert set(stripped.keys()) == set(policy_state_dict.keys())
    # tensor values are the same objects (no clone/detach mangling).
    assert stripped["encoder.weight"] is policy_state_dict["encoder.weight"]
