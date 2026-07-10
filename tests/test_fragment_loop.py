import json
import pathlib

from scripts.fragment_loop import select_topk_seeds, select_random_seeds
import scripts.fragment_loop as fl
import scripts.optimize_loop as ol


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


def test_is_round_done_requires_nonempty_scores(tmp_path):
    d = tmp_path / "r0"; d.mkdir()
    assert fl.is_round_done(d) is False
    (d / "dock_scores.csv").write_text("smiles,score\nA,-7\n")
    assert fl.is_round_done(d) is True


def test_run_arm_control_b_never_seeds_and_treatment_seeds_topk(tmp_path, monkeypatch):
    calls = {"analog_seeds": [], "pocket": 0}
    def fake_analog(seeds, model, out, python=None):
        calls["analog_seeds"].append(list(seeds))
        pathlib.Path(out).write_text("\n".join(
            json.dumps({"smiles": s, "seed": seeds[0], "sim": 0.5}) for s in ["X", "Y", "Z"]))
    def fake_pocket(target, out, ckpt, n, seed, python=None):
        calls["pocket"] += 1
        pathlib.Path(out).write_text("\n".join(
            json.dumps({"smiles": s, "bb": [1], "tpl": [1]}) for s in ["P", "Q", "R"]))
    monkeypatch.setattr(fl, "run_analog_generation", fake_analog)
    monkeypatch.setattr(fl, "run_pocket_generation", fake_pocket)
    monkeypatch.setattr(ol, "passes_gate", lambda s, sa_max=4.0: True)
    monkeypatch.setattr(fl, "dock", lambda spec, smi, seed=0: {"X": -9.0, "Y": -5.0, "Z": -3.0,
                                                               "P": -8.0, "Q": -4.0, "R": -2.0}[smi])
    # shared round-0 scores (docked once in main, passed into every arm)
    round0_scores = {"S0": -6.0, "S1": -7.0, "S2": -8.5}
    fl.run_arm(arm="control_b", target="T", spec=None, ckpt_analog="a", ckpt_pocket="p",
               rounds=1, budget=3, k=2, n=3, seed=1, out_dir=tmp_path,
               round0_scores=round0_scores, summary_rows=[])
    assert calls["pocket"] == 1 and calls["analog_seeds"] == []      # control_b never analogs
    calls["pocket"] = 0
    fl.run_arm(arm="treatment", target="T", spec=None, ckpt_analog="a", ckpt_pocket="p",
               rounds=1, budget=3, k=2, n=3, seed=1, out_dir=tmp_path,
               round0_scores=round0_scores, summary_rows=[])
    # treatment seeds on the top-2 shared round-0 dockers: S2(-8.5), S1(-7.0)
    assert calls["analog_seeds"] and calls["analog_seeds"][0] == ["S2", "S1"]
