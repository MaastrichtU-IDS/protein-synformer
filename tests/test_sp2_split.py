from scripts.make_sp2_split import split_targets


def test_splits_are_protein_disjoint_and_cover_all():
    targets = [f"P{i:05d}" for i in range(1000)]
    train, val, test = split_targets(targets)
    assert train and val and test
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == set(targets)


def test_split_is_deterministic():
    targets = [f"P{i:05d}" for i in range(1000)]
    assert split_targets(targets) == split_targets(list(reversed(targets)))
