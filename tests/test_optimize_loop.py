# tests/test_optimize_loop.py
from scripts.optimize_loop import (
    gate_and_dedup, dock_budget, select_winners, next_weights,
)


def _rec(smi, bb, tpl):
    return {"smiles": smi, "bb": bb, "tpl": tpl}


def test_gate_and_dedup_drops_invalid_and_dupes(monkeypatch):
    import scripts.optimize_loop as ol
    monkeypatch.setattr(ol, "passes_gate", lambda s, sa_max=4.0: s != "BAD")
    recs = [_rec("A", [1], [1]), _rec("A", [1], [1]), _rec("BAD", [2], [2]), _rec("C", [3], [3])]
    out = gate_and_dedup(recs)
    assert [r["smiles"] for r in out] == ["A", "C"]


def test_dock_budget_excludes_nan_and_respects_budget():
    recs = [_rec(s, [1], [1]) for s in ["A", "B", "C"]]
    scores = {"A": -7.0, "B": float("nan"), "C": -5.0}
    fn = lambda spec, smi, seed=0: scores[smi]
    got = dock_budget(recs, spec=None, dock_fn=fn, budget=3, seed=1, max_workers=2)
    assert got == {"A": -7.0, "C": -5.0}  # nan dropped
    got2 = dock_budget(recs, spec=None, dock_fn=fn, budget=1, seed=1, max_workers=2)
    assert set(got2) == {"A"}  # budget honoured (first gated candidate)


def test_select_winners_takes_most_negative():
    assert select_winners({"A": -7.0, "B": -3.0, "C": -9.0}, 2) == ["C", "A"]


def test_next_weights_promotes_winner_building_blocks():
    recs = [_rec("A", [1], [1]), _rec("B", [1], [1]), _rec("C", [9], [9]), _rec("D", [9], [9])]
    w = next_weights(["A", "B"], recs, w_max=5.0)
    assert w["bb"].get("1", 1.0) > 1.0
    assert "9" not in w["bb"]
