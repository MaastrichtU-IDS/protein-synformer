import torch
from torch import nn

from synformer.data.common import ProjectionBatch
from .base import BaseEncoder, EncoderOutput


class ProteinTransformerEncoder(BaseEncoder):
    """Projects ESM per-residue embeddings then applies self-attention so residues
    interact before the decoder cross-attends. Honors the padding mask."""

    def __init__(self, d_model: int = 768, d_protein: int = 1152, nhead: int = 8,
                 dim_feedforward: int = 2048, num_layers: int = 2,
                 dropout: float = 0.1, output_norm: bool = False):
        super().__init__()
        self._dim = d_model
        self.proj = nn.Linear(d_protein, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(
            layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if output_norm else None,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: ProjectionBatch) -> EncoderOutput:
        if "protein_embeddings" not in batch:
            raise ValueError("protein_embeddings must be in batch")
        mask = batch.get("protein_padding_mask", None)  # (B, L) True = pad
        code = self.enc(self.proj(batch["protein_embeddings"]), src_key_padding_mask=mask)
        return EncoderOutput(code, mask)
