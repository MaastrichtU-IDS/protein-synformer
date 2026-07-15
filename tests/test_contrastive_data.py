from scripts.contrastive_data import binder_label, make_within_family_triples


def test_binder_label():
    assert binder_label(12.5) == "bind"
    assert binder_label(11.0) == "non"
    assert binder_label(11.7) is None   # ambiguous middle dropped


def test_make_within_family_triples_train_only_and_within_family():
    rows = [{"smiles": "X", "gene": "MAPK1", "kiba": 12.5},
            {"smiles": "X", "gene": "MAPK3", "kiba": 11.0},
            {"smiles": "X", "gene": "CSNK1D", "kiba": 12.5},
            {"smiles": "X", "gene": "CSNK1E", "kiba": 11.0}]
    gene2fam = {"MAPK1": "MAPK", "MAPK3": "MAPK", "CSNK1D": "CSNK1", "CSNK1E": "CSNK1"}
    tr = make_within_family_triples(rows, gene2fam, train_fams={"MAPK"})
    assert ("X", "MAPK1", "MAPK3", "MAPK") in tr
    assert all(t[3] == "MAPK" for t in tr)
    assert not any("CSNK1" in t[1] or "CSNK1" in t[2] for t in tr)
