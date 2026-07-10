import torch
from synformer.data.common import TokenType, featurize_stack_actions


def test_featurize_stack_actions_marks_reaction_and_reactant_tokens():
    # one reaction at step 1, one reactant (building block) at step 2
    class FakeFpindex:
        class _opt: dim = 4
        fp_option = _opt()
        def __getitem__(self, i):  # returns (mol, fp)
            import numpy as np
            return None, np.ones(4, dtype="float32") * i
    feats = featurize_stack_actions(
        mol_idx_seq=[None, 7], rxn_idx_seq=[3, None], end_token=False, fpindex=FakeFpindex()
    )
    assert feats["token_types"][0] == TokenType.START
    assert feats["token_types"][1] == TokenType.REACTION
    assert feats["rxn_indices"][1] == 3
    assert feats["token_types"][2] == TokenType.REACTANT
    assert torch.allclose(feats["reactant_fps"][2], torch.full((4,), 7.0))
