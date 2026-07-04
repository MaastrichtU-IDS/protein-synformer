from synformer.chem.mol import Molecule
from synformer.eval.generation import (
    internal_diversity,
    novelty,
    scaffold_diversity,
    uniqueness,
)
from synformer.eval.synthesizability import route_lengths, sa_score


def test_uniqueness_canonical_dedup():
    # "CCO" and "OCC" are the same molecule; benzene is distinct -> 2 unique / 3
    assert abs(uniqueness(["CCO", "OCC", "c1ccccc1"]) - 2 / 3) < 1e-9


def test_uniqueness_ignores_invalid():
    assert uniqueness(["CCO", "not_a_smiles"]) == 1.0


def test_novelty_against_reference():
    # ethanol is in the reference, benzene is not -> 0.5 novel
    assert novelty(["CCO", "c1ccccc1"], ["OCC"]) == 0.5


def test_internal_diversity_identical_is_zero():
    assert internal_diversity([Molecule("CCO"), Molecule("CCO")]) == 0.0


def test_internal_diversity_distinct_is_positive():
    assert internal_diversity([Molecule("CCO"), Molecule("c1ccccc1")]) > 0.5


def test_scaffold_diversity_shared_scaffold():
    # both are substituted benzenes -> same Murcko scaffold -> 1 unique / 2
    assert scaffold_diversity([Molecule("Cc1ccccc1"), Molecule("CCc1ccccc1")]) == 0.5


def test_sa_score_in_valid_range():
    assert 1.0 <= sa_score(Molecule("CCO")) <= 10.0


def test_route_lengths_reads_cnt_rxn():
    infos = {"T1": {0: {"cnt_rxn": 3}, 1: {"cnt_rxn": 5}}}
    assert route_lengths(infos) == [3, 5]
