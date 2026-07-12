"""DPO for the pocket-conditioned SynFormer: preference loss on generation routes."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def dpo_loss(llpi_w, llpi_l, llref_w, llref_l, beta: float = 0.1):
    """Standard DPO loss (mean over pairs). ll* are per-pair total log-likelihoods."""
    pi_margin = llpi_w - llpi_l
    ref_margin = llref_w - llref_l
    return -F.logsigmoid(beta * (pi_margin - ref_margin)).mean()


def routes_from_result(result) -> list[dict]:
    """Slice per-molecule route tensors from a batched GenerateResult (CPU)."""
    n = result.token_types.size(0)
    out = []
    for i in range(n):
        out.append({
            "token_types": result.token_types[i:i+1].cpu(),
            "rxn_indices": result.rxn_indices[i:i+1].cpu(),
            "reactant_fps": result.reactant_fps[i:i+1].cpu(),
            "token_padding_mask": result.token_padding_mask[i:i+1].cpu(),
        })
    return out
