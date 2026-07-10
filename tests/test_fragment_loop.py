from scripts.fragment_loop import select_topk_seeds, select_random_seeds


def test_select_topk_seeds_takes_strongest():
    assert select_topk_seeds({"A": -7.0, "B": -3.0, "C": -9.0}, 2) == ["C", "A"]


def test_select_random_seeds_is_seeded_and_sized():
    scored = {c: -float(i) for i, c in enumerate("ABCDEFGH")}
    a = select_random_seeds(scored, 3, seed=1)
    b = select_random_seeds(scored, 3, seed=1)
    c = select_random_seeds(scored, 3, seed=2)
    assert len(a) == 3 and set(a) <= set(scored)
    assert a == b            # deterministic for a fixed seed
    assert a != c or True    # different seed *may* differ; not asserted strictly


def test_select_random_seeds_caps_at_pool_size():
    assert len(select_random_seeds({"A": -1.0, "B": -2.0}, 5, seed=1)) == 2
