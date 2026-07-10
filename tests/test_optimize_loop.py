# tests/test_optimize_loop.py
import pytest

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


def test_next_weights_denominator_is_docked_pool_not_gated_pool():
    # Docked pool (what actually got dock scores this round): A is the sole winner and
    # is the only docked record carrying building block 1; B/C/D carry building block 2.
    docked_recs = [_rec("A", [1], [1]), _rec("B", [2], [2]),
                   _rec("C", [2], [2]), _rec("D", [2], [2])]
    # E is gated but was NEVER docked this round (e.g. dropped by the dock budget cutoff).
    # It shares the winner's building block (1), so folding it into the enrichment
    # denominator dilutes bb=1's pool frequency (1/4 -> 2/5) even though E never competed
    # in the docking round at all.
    gated_but_undocked = _rec("E", [1], [1])

    w_docked = next_weights(["A"], docked_recs, w_max=5.0)
    w_gated = next_weights(["A"], docked_recs + [gated_but_undocked], w_max=5.0)

    # Correct denominator (docked pool only): f_pool(bb=1) = 1/4 = 0.25
    expected_docked_ratio = 1.0 / (0.25 + 1e-3)
    assert w_docked["bb"]["1"] == pytest.approx(expected_docked_ratio, rel=1e-6)

    # Buggy denominator (gated pool incl. undocked E): f_pool(bb=1) = 2/5 = 0.4, a lower
    # (diluted) ratio -- proving that including undocked records changes the result and
    # that the fix (restricting to the docked pool) is what avoids the dilution.
    expected_gated_ratio = 1.0 / (0.4 + 1e-3)
    assert w_gated["bb"]["1"] == pytest.approx(expected_gated_ratio, rel=1e-6)
    assert w_gated["bb"]["1"] < w_docked["bb"]["1"]


def test_run_arm_passes_docked_subset_to_next_weights(tmp_path, monkeypatch):
    # End-to-end call-site check: run_arm must call next_weights with only the docked
    # subset of `recs`, even though the gated pool (recs) contains an extra molecule
    # ("D") that passed the gate but was never docked (excluded by the budget cutoff).
    captured_pools = []
    real_next_weights = ol.next_weights

    def spy_next_weights(winners, pool, **kw):
        captured_pools.append([r["smiles"] for r in pool])
        return real_next_weights(winners, pool, **kw)

    monkeypatch.setattr(ol, "next_weights", spy_next_weights)

    def fake_gen(ckpt, target, weights_path, n, seed, out_path, python=None):
        pathlib.Path(out_path).write_text(
            "\n".join(json.dumps({"smiles": s, "bb": [1], "tpl": [1]}) for s in ["A", "B", "C", "D"]))
    monkeypatch.setattr(ol, "run_generation", fake_gen)
    monkeypatch.setattr(ol, "passes_gate", lambda s, sa_max=4.0: True)
    # dock_budget only docks the first `budget` of the gated records; "D" is gated but
    # never docked because budget=3 truncates the pool of 4 candidates before scoring.
    monkeypatch.setattr(ol, "dock", lambda spec, smi, seed=0: {"A": -9.0, "B": -5.0, "C": -3.0}[smi])

    ol.run_arm(ckpt="x", target="T", arm="enrich", spec=None, rounds=1, budget=3, n=4, k=1,
               seed=1, out_dir=tmp_path)

    assert len(captured_pools) == 1
    assert "D" not in captured_pools[0]
    assert set(captured_pools[0]) == {"A", "B", "C"}


import json, pathlib
import scripts.optimize_loop as ol


def test_is_round_done_requires_nonempty_scores(tmp_path):
    d = tmp_path / "r0"; d.mkdir()
    assert ol.is_round_done(d) is False
    (d / "dock_scores.csv").write_text("")
    assert ol.is_round_done(d) is False
    (d / "dock_scores.csv").write_text("smiles,score\nA,-7\n")
    assert ol.is_round_done(d) is True


def test_run_arm_resumes_and_uniform_uses_no_weights(tmp_path, monkeypatch):
    # stub generation: write a fixed candidate file; record the weights arg seen each round
    seen_weights = []
    def fake_gen(ckpt, target, weights_path, n, seed, out_path, python=None):
        seen_weights.append(pathlib.Path(weights_path).name if weights_path not in (None, "NONE") else "NONE")
        pathlib.Path(out_path).write_text(
            "\n".join(json.dumps({"smiles": s, "bb": [1], "tpl": [1]}) for s in ["A", "B", "C"]))
    monkeypatch.setattr(ol, "run_generation", fake_gen)
    monkeypatch.setattr(ol, "passes_gate", lambda s, sa_max=4.0: True)
    monkeypatch.setattr(ol, "dock", lambda spec, smi, seed=0: {"A": -9.0, "B": -5.0, "C": -3.0}[smi])
    ol.run_arm(ckpt="x", target="T", arm="uniform", spec=None, rounds=2, budget=3, n=3, k=1,
               seed=1, out_dir=tmp_path)
    assert seen_weights == ["NONE", "NONE"]  # uniform never enriches
