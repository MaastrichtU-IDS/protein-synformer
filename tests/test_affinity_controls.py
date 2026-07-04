from scripts.affinity_controls import derangement, foreign_ligands


def test_derangement_has_no_fixed_points():
    perm = derangement(50, seed=0)
    assert sorted(perm) == list(range(50))
    assert all(perm[i] != i for i in range(50))


def test_foreign_ligands_excludes_own():
    gt = {"A": ["c1ccccc1"], "B": ["CCO"], "C": ["CCN"]}
    picks = foreign_ligands("A", gt, r=2, seed=0)
    assert len(picks) == 2
    assert "c1ccccc1" not in picks
