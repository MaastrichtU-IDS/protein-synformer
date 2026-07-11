import numpy as np
import pandas as pd
from scripts.candidate_agreement import spearman_by_target, hacking_percentile, selection_overlap


def test_spearman_agree_vs_disagree():
    agree = pd.DataFrame({"target": ["A"] * 5, "molecule": list("abcde"),
                          "smina": [-9, -7, -5, -3, -1], "boltz": [-9, -7, -5, -3, -1]})
    disagree = agree.copy(); disagree["boltz"] = [-1, -3, -5, -7, -9]  # inverted
    assert spearman_by_target(agree)["A"] > 0.99
    assert spearman_by_target(disagree)["A"] < -0.99


def test_hacking_percentile_low_when_smina_top_are_boltz_weak():
    # smina-strongest (m0,m1) are Boltz-weakest -> low percentile
    df = pd.DataFrame({"target": ["A"] * 5, "molecule": [f"m{i}" for i in range(5)],
                       "smina": [-9, -8, -5, -3, -1], "boltz": [-1, -2, -5, -8, -9]})
    p = hacking_percentile(df, k=2)["A"]
    assert p < 0.4          # smina-top are near the bottom of Boltz


def test_selection_overlap_jaccard():
    # smina and boltz pick disjoint top-2 -> jaccard 0
    df = pd.DataFrame({"target": ["A"] * 4, "molecule": list("abcd"),
                       "smina": [-9, -8, -2, -1], "boltz": [-1, -2, -8, -9]})
    o = selection_overlap(df, k=2)["A"]
    assert o["smina_vs_boltz"] == 0.0
