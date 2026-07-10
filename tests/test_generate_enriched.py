import json
from scripts.generate_enriched import load_weights, stacks_to_records


class FakeStack:
    def __init__(self, smi, mol_idx, rxn_idx):
        self._smi, self._mol_idx, self._rxn_idx = smi, mol_idx, rxn_idx
    def get_one_top(self):
        class M: smiles = self._smi
        return M()
    def get_mol_idx_seq(self): return self._mol_idx
    def get_rxn_idx_seq(self): return self._rxn_idx
    def get_stack_depth(self): return 1


def test_load_weights_none(tmp_path):
    assert load_weights(None) is None
    assert load_weights("NONE") is None


def test_load_weights_parses_int_keys(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"bb": {"5": 2.0}, "tpl": {"3": 1.5}}))
    w = load_weights(str(p))
    assert w.bb == {5: 2.0} and w.tpl == {3: 1.5}


def test_stacks_to_records_extracts_indices():
    stacks = [FakeStack("CCO", [10, None, 11], [None, 2, None])]
    recs = stacks_to_records(stacks)
    assert recs[0]["smiles"] == "CCO"
    assert sorted(recs[0]["bb"]) == [10, 11]
    assert recs[0]["tpl"] == [2]
