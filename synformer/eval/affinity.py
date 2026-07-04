"""Predicted binding-affinity scoring (the paper's biggest gap).

Productionizes the DeepPurpose approach from `Sample.ipynb`: a pretrained
drug-target interaction model (MPNN drug encoder + CNN protein encoder, trained
on DAVIS) scores (SMILES, protein-sequence) pairs. Higher = stronger predicted
binding (DAVIS labels are pKd). This is a fast, structure-free ML scorer suited
to our sequence-only data; a docking backend can be added later behind the same
`predict_affinity` interface.
"""
from collections.abc import Sequence

import numpy as np


def load_scorer(model_name: str = "MPNN_CNN_DAVIS"):
    """Load a pretrained DeepPurpose DTI model (downloads weights on first use)."""
    from DeepPurpose import DTI as models
    return models.model_pretrained(model=model_name)


def predict_affinity(model, smiles: Sequence[str], sequences: Sequence[str]) -> np.ndarray:
    """Predict binding affinity for paired (drug SMILES, protein sequence) lists.

    `smiles` and `sequences` must be equal length and aligned. Returns one score
    per pair. Encodings (MPNN drug / CNN target) match the pretrained model.
    """
    import DeepPurpose.utils as utils
    if len(smiles) != len(sequences):
        raise ValueError("smiles and sequences must be the same length")
    X = utils.data_process(
        list(smiles), list(sequences), np.zeros(len(smiles)),
        "MPNN", "CNN", split_method="no_split",
    )
    return np.asarray(model.predict(X))
