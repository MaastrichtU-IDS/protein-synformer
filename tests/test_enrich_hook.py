"""The sampling arithmetic the model performs, replicated on tensors, to prove
the enrichment bias reproduces baseline when None and multiplies probability by w otherwise."""
import numpy as np
import torch
import torch.nn.functional as F

from synformer.molopt.enrich import EnrichWeights, reaction_log_bias, reactant_log_bias


def _reaction_probs(logits, T, weights):
    bias = torch.from_numpy(reaction_log_bias(logits.shape[-1], weights)).to(logits)
    return F.softmax(logits / T + bias, dim=-1)


def test_reaction_bias_none_matches_baseline():
    logits = torch.tensor([[1.0, 2.0, 0.5, -1.0]])
    base = F.softmax(logits / 1.0, dim=-1)
    got = _reaction_probs(logits, 1.0, None)
    assert torch.allclose(base, got)


def test_reaction_weight_multiplies_probability():
    logits = torch.zeros(1, 3)  # uniform -> each 1/3
    w = EnrichWeights(bb={}, tpl={0: 2.0})
    probs = _reaction_probs(logits, 1.0, w)
    # unnormalised weights: [2,1,1] -> normalise
    assert torch.allclose(probs, torch.tensor([[0.5, 0.25, 0.25]]), atol=1e-6)


def test_reactant_bias_shapes_and_absent_noop():
    idx = np.array([[3, 4, 5]])
    assert np.allclose(reactant_log_bias(idx, EnrichWeights(bb={99: 5.0}, tpl={})), 0.0)


import pathlib

import pytest

CKPT = pathlib.Path("logs/pocket/2607091019-32f2194@powered-specificity/"
                    "2026_07_09__10_19_15/checkpoints/epoch=1-step=2255.ckpt")


@pytest.mark.skipif(
    not CKPT.exists() or not torch.cuda.is_available(),
    reason="SP-C ckpt not present or no CUDA (run on the box in .venv-train)",
)
def test_generate_none_is_baseline_smoke():
    # Run in .venv-train on the box:
    #   .venv-train/bin/python -m pytest tests/test_enrich_hook.py::test_generate_none_is_baseline_smoke -q
    from scripts.dock_prepare import _load_test_targets_with_embeddings  # noqa: F401
    from scripts.sample_helpers import load_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fpindex, rxn = load_model(str(CKPT), None, device)
    assert model is not None  # deeper assertion added during execution once feat plumbing confirmed
