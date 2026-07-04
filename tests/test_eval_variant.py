from scripts.eval_variant import summarize_infos


def test_summarize_infos_basic():
    # one protein, two generated molecules, one ground-truth
    infos = {"T1": {0: {"smiles": "CCO", "cnt_rxn": 1},
                    1: {"smiles": "c1ccccc1", "cnt_rxn": 2}}}
    gt = {"T1": ["CCO"]}
    m = summarize_infos(infos, gt, repeat=4)
    assert m["n_proteins"] == 1
    assert abs(m["validity"] - 0.5) < 1e-9        # 2 built / repeat 4
    assert 0.0 <= m["novelty"] <= 1.0
    assert 0.49 <= m["sim_best_pair_mean"] <= 0.51  # mean of [1.0, 0.0] = 0.5
