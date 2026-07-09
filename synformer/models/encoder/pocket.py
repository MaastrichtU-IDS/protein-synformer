"""SE(3)-invariant 3D binding-pocket encoder: residue-type embeddings + a pairwise
CA-distance attention bias (RBF-expanded), a few transformer layers, ReZero output-gate.
Produces conditioning tokens the decoder cross-attends to (same interface as the sequence
encoders). Invariant to rigid transforms because it uses only pairwise distances."""
from __future__ import annotations

import torch
from torch import nn

from synformer.data.common import ProjectionBatch
from .base import BaseEncoder, EncoderOutput


class _BiasedLayer(nn.Module):
    """Transformer encoder layer whose self-attention takes an additive per-head bias
    (from pocket distances) and a key-padding mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias, key_padding_mask):
        h = self.norm1(x)
        # attn_bias: (B*heads, N, N) additive float mask; key_padding_mask: (B, N) True=pad
        a, _ = self.attn(h, h, h, attn_mask=attn_bias, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class PocketEncoder(BaseEncoder):
    def __init__(self, d_model: int = 768, n_restype: int = 21, n_layers: int = 4,
                 n_heads: int = 8, n_rbf: int = 32, r_max: float = 15.0, dropout: float = 0.1):
        super().__init__()
        self._dim = d_model
        self._n_heads = n_heads
        self.type_emb = nn.Embedding(n_restype, d_model)          # 20 AA + pad idx 20
        self.register_buffer("rbf_centers", torch.linspace(0.0, r_max, n_rbf))
        self.rbf_gamma = (n_rbf / r_max) ** 2 * 0.5
        self.dist_to_bias = nn.Linear(n_rbf, n_heads)
        self.layers = nn.ModuleList([_BiasedLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out_gate = nn.Parameter(torch.tensor(0.05))          # ReZero (SP2 collapse fix)

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: ProjectionBatch) -> EncoderOutput:
        for k in ("pocket_restype", "pocket_ca", "pocket_padding_mask"):
            if k not in batch:
                raise ValueError(f"{k} must be in batch")
        rt = batch["pocket_restype"]                              # (B,N) long
        ca = batch["pocket_ca"].to(torch.float32)                # (B,N,3)
        mask = batch["pocket_padding_mask"]                      # (B,N) bool, True=pad
        B, N = rt.shape

        x = self.type_emb(rt)                                     # (B,N,d)
        # SE(3)-invariant geometry: pairwise CA distances -> RBF -> per-head additive bias
        d = torch.cdist(ca, ca)                                   # (B,N,N)
        rbf = torch.exp(-self.rbf_gamma * (d.unsqueeze(-1) - self.rbf_centers) ** 2)  # (B,N,N,n_rbf)
        bias = self.dist_to_bias(rbf)                             # (B,N,N,heads)
        bias = bias.permute(0, 3, 1, 2).reshape(B * self._n_heads, N, N)             # (B*heads,N,N)

        for layer in self.layers:
            x = layer(x, attn_bias=bias, key_padding_mask=mask)
        code = self.out_gate * self.norm(x)
        return EncoderOutput(code, mask)
