import torch
from synformer.data.projection_dataset_new import Collater


def _example(prot_len, n_tokens=3, fp_dim=2048):
    return {
        "protein_embeddings": torch.randn(prot_len, 1152),
        "protein_padding_mask": torch.zeros(prot_len, dtype=torch.bool),
        "token_types": torch.ones(n_tokens, dtype=torch.long),
        "rxn_indices": torch.zeros(n_tokens, dtype=torch.long),
        "reactant_fps": torch.zeros(n_tokens, fp_dim),
        "token_padding_mask": torch.zeros(n_tokens, dtype=torch.bool),
        "mol_seq": [], "rxn_seq": [],
    }


def test_collater_emits_protein_padding_mask():
    batch = Collater(max_protein_len=10, max_num_tokens=4)([_example(4), _example(7)])
    m = batch["protein_padding_mask"]
    assert m.shape == (2, 10)
    assert not m[0, :4].any() and m[0, 4:].all()   # first 4 real, rest padded
    assert not m[1, :7].any() and m[1, 7:].all()
