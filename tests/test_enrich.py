from synformer.molopt.enrich import (
    EnrichWeights, molecule_index_sets, compute_enrichment_weights,
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
