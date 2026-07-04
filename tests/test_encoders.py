import torch
from synformer.models.encoder import get_encoder


def _batch(B=2, L=5):
    return {
        "protein_embeddings": torch.randn(B, L, 1152),
        "protein_padding_mask": torch.zeros(B, L, dtype=torch.bool),
    }


def test_protein_masked_shapes_and_mask():
    enc = get_encoder("protein_masked", {"d_model": 32, "d_protein": 1152, "hidden_dim": 64})
    out = enc(_batch())
    assert out.code.shape == (2, 5, 32)
    assert out.code_padding_mask.shape == (2, 5)
    assert enc.dim == 32


def test_protein_transformer_shapes_and_mask():
    enc = get_encoder("protein_transformer", {
        "d_model": 32, "d_protein": 1152, "nhead": 4,
        "dim_feedforward": 64, "num_layers": 2,
    })
    b = _batch(B=2, L=5)
    b["protein_padding_mask"][1, 3:] = True   # protein 2 has 3 real residues
    out = enc(b)
    assert out.code.shape == (2, 5, 32)
    assert out.code_padding_mask.shape == (2, 5)
    assert bool(out.code_padding_mask[1, 4])   # padding preserved


def test_protein_intermediate_returns_latent_mask():
    enc = get_encoder("protein_intermediate", {
        "d_model": 32, "d_protein": 1152, "nhead": 4,
        "dim_feedforward": 64, "num_layers": 1, "num_latents": 8,
    })
    out = enc(_batch(B=2, L=5))
    assert out.code.shape == (2, 8, 32)          # (B, num_latents, d_model)
    assert out.code_padding_mask.shape == (2, 8)
    assert not out.code_padding_mask.any()       # latents are never padding
