from torch import nn
from synformer.data.common import ProjectionBatch

from .base import BaseEncoder, EncoderOutput


class ProteinEncoder(BaseEncoder):
    def __init__(self,
                 d_model: int = 512,
                 d_protein: int = 1152,
                 emit_mask: bool = False):
        super().__init__()
        self._dim = d_model
        self._emit_mask = emit_mask
        # Project protein embeddings from d_protein (e.g. 1152) to d_model (e.g. 512 or 768)
        self.enc = nn.Linear(d_protein, d_model)

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: ProjectionBatch):
        if "protein_embeddings" not in batch:
            raise ValueError("protein_embeddings must be in batch")
        # Shape: (batch_size, seq_len, d_protein) -> (batch_size, seq_len, d_model)
        code = self.enc(batch["protein_embeddings"])
        # emit_mask=False -> study's original behavior (decoder attends over padded residues);
        # emit_mask=True -> forward the padding mask (SP2 ablation: linear + mask).
        code_padding_mask = batch.get("protein_padding_mask", None) if self._emit_mask else None
        return EncoderOutput(code, code_padding_mask)
