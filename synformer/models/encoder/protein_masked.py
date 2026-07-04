from torch import nn

from synformer.data.common import ProjectionBatch
from .base import BaseEncoder, EncoderOutput


class ProteinMaskedEncoder(BaseEncoder):
    """Per-residue MLP projection that also forwards a padding mask so the decoder
    cross-attention ignores zero-padded residues (the plain ProteinEncoder does not)."""

    def __init__(self, d_model: int = 768, d_protein: int = 1152,
                 hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self._dim = d_model
        self.enc = nn.Sequential(
            nn.Linear(d_protein, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: ProjectionBatch) -> EncoderOutput:
        if "protein_embeddings" not in batch:
            raise ValueError("protein_embeddings must be in batch")
        code = self.enc(batch["protein_embeddings"])
        return EncoderOutput(code, batch.get("protein_padding_mask", None))
