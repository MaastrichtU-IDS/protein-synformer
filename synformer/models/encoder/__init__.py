from .base import BaseEncoder, NoEncoder
from .graph import GraphEncoder
from .smiles import SMILESEncoder
from .protein import ProteinEncoder
from .protein_intermediate import ProteinIntermediateEncoder
from .protein_masked import ProteinMaskedEncoder
from .protein_transformer import ProteinTransformerEncoder


def get_encoder(t: str, cfg) -> BaseEncoder:
    if t == "smiles":
        return SMILESEncoder(**cfg)
    elif t == "graph":
        return GraphEncoder(**cfg)
    elif t == "protein":
        return ProteinEncoder(**cfg)
    elif t == "protein_intermediate":
        return ProteinIntermediateEncoder(**cfg)
    elif t == "protein_masked":
        return ProteinMaskedEncoder(**cfg)
    elif t == "protein_transformer":
        return ProteinTransformerEncoder(**cfg)
    elif t == "none":
        return NoEncoder(**cfg)
    else:
        raise ValueError(f"Unknown encoder type: {t}")
