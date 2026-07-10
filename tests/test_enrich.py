import numpy as np
from synformer.molopt.enrich import (
    EnrichWeights, molecule_index_sets, compute_enrichment_weights,
    reaction_log_bias, reactant_log_bias, passes_gate, sa_score,
)


def test_molecule_index_sets_drops_none_and_sentinel():
    bb, tpl = molecule_index_sets([5, None, 7, -1], [None, 2, None, 3])
    assert bb == frozenset({5, 7})
    assert tpl == frozenset({2, 3})


def test_enrichment_weight_is_presence_fraction_ratio_clipped():
    # BB 1 appears in 2/2 winners but 1/4 pool -> ratio (1.0)/(0.25)=4.0
    # BB 9 appears in 0 winners -> absent from weights (no promotion)
    winners = [(frozenset({1}), frozenset()), (frozenset({1}), frozenset())]
    pool = [
        (frozenset({1}), frozenset()),
        (frozenset({9}), frozenset()), (frozenset({9}), frozenset()), (frozenset({9}), frozenset()),
    ]
    w = compute_enrichment_weights(winners, pool, w_max=5.0, eps=1e-3)
    assert abs(w.bb[1] - 4.0) < 2e-2
    assert 9 not in w.bb


def test_enrichment_weight_clipped_to_wmax_and_floored_at_one():
    # BB 1: 1.0 winners / tiny pool -> huge ratio -> clipped to w_max
    winners = [(frozenset({1}), frozenset())]
    pool = [(frozenset({2}), frozenset())]  # BB 1 absent from pool
    w = compute_enrichment_weights(winners, pool, w_max=5.0, eps=1e-3)
    assert w.bb[1] == 5.0  # clipped
    # a BB that is rarer in winners than pool is floored at 1.0 (enrichment only promotes)


def test_empty_winners_gives_uniform_weights():
    w = compute_enrichment_weights([], [(frozenset({1}), frozenset())])
    assert w.bb == {} and w.tpl == {}


def test_reaction_log_bias_none_is_zero():
    b = reaction_log_bias(5, None)
    assert b.shape == (5,) and np.allclose(b, 0.0)


def test_reaction_log_bias_sets_log_weight_for_enriched_templates():
    w = EnrichWeights(bb={}, tpl={2: np.e})  # log(e)=1
    b = reaction_log_bias(4, w)
    assert np.allclose(b, [0.0, 0.0, 1.0, 0.0])


def test_reactant_log_bias_present_index_gets_weight_absent_is_zero():
    # retrieved BBs: rows are batch, cols are the top-k retrieved for the step
    idx = np.array([[10, 11], [12, 10]])
    w = EnrichWeights(bb={10: np.e}, tpl={})  # only BB 10 up-weighted
    b = reactant_log_bias(idx, w)
    assert np.allclose(b, [[1.0, 0.0], [0.0, 1.0]])


def test_reactant_log_bias_absent_bb_has_no_effect():
    # BB 99 is up-weighted but never retrieved -> bias stays all-zero
    idx = np.array([[10, 11]])
    w = EnrichWeights(bb={99: 5.0}, tpl={})
    assert np.allclose(reactant_log_bias(idx, w), 0.0)


def test_gate_rejects_invalid_smiles():
    assert passes_gate("not_a_smiles") is False


def test_gate_rejects_too_small():
    assert passes_gate("CCO") is False  # 3 heavy atoms < MIN_HEAVY_ATOMS


def test_gate_rejects_disallowed_element():
    # a boron-containing molecule large enough otherwise
    assert passes_gate("B1OC2=CC=CC=C2O1" * 1) is False


def test_gate_accepts_drug_like():
    # ibuprofen: 15 heavy atoms, CHO only, low SA
    assert passes_gate("CC(C)Cc1ccc(cc1)C(C)C(=O)O") is True


def test_sa_score_finite_for_valid():
    assert sa_score("CC(C)Cc1ccc(cc1)C(C)C(=O)O") < 4.0
