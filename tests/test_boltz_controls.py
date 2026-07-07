import json

import numpy as np
import pandas as pd

from scripts.boltz_controls import enumerate_control_cells, cell_done, stem_for
from scripts.boltz_controls_analyze import discrimination_auroc


def _write_inputs(tmp_path):
    d = {"hits": [], "proteins": [
        {"target_id": "A", "sequence": "AAAA"},
        {"target_id": "B", "sequence": "CCCC"}]}
    p = tmp_path / "inputs.json"
    json.dump(d, open(p, "w"))
    return str(p)


def _write_dock(tmp_path):
    rows = [
        # A: 2 known, 1 random (own pocket)
        {"target": "A", "pocket": "A", "molecule": "CCO", "source": "known", "score": -7.0},
        {"target": "A", "pocket": "A", "molecule": "CCC", "source": "known", "score": -8.0},
        {"target": "A", "pocket": "A", "molecule": "CN", "source": "random", "score": -4.0},
        # cross-pocket rows must be ignored
        {"target": "A", "pocket": "B", "molecule": "CCO", "source": "known", "score": -1.0},
        # candidate rows must be ignored
        {"target": "A", "pocket": "A", "molecule": "CCCC", "source": "candidate", "score": -9.0},
        # B: 1 known, 1 random
        {"target": "B", "pocket": "B", "molecule": "CO", "source": "known", "score": -6.0},
        {"target": "B", "pocket": "B", "molecule": "CCN", "source": "random", "score": -3.0},
    ]
    p = tmp_path / "dock.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def test_enumerate_control_cells_own_pocket_known_and_random(tmp_path):
    cells = enumerate_control_cells(_write_dock(tmp_path), _write_inputs(tmp_path))
    # A: 2 known + 1 random ; B: 1 known + 1 random = 5 cells (candidate + cross-pocket excluded)
    assert len(cells) == 5
    classes = sorted((c["target"], c["class"]) for c in cells)
    assert classes.count(("A", "known")) == 2
    assert classes.count(("A", "random")) == 1
    # own-pocket sequence attached
    a_cell = [c for c in cells if c["target"] == "A"][0]
    assert a_cell["sequence"] == "AAAA"
    b_cell = [c for c in cells if c["target"] == "B"][0]
    assert b_cell["sequence"] == "CCCC"


def test_stem_stable_and_target_scoped():
    # same SMILES, different target -> different stem (different co-fold)
    assert stem_for("A", "known", "CCO") == stem_for("A", "known", "CCO")
    assert stem_for("A", "known", "CCO") != stem_for("B", "known", "CCO")


def test_cell_done_keys_on_target_and_smiles(tmp_path):
    csv = tmp_path / "c.csv"
    pd.DataFrame([{"target": "A", "smiles": "CCO", "class": "known",
                   "affinity_pred": -1.0, "binder_prob": 0.7, "ligand_iptm": 0.9}]).to_csv(csv, index=False)
    assert cell_done(str(csv), "A", "CCO") is True
    assert cell_done(str(csv), "A", "CCC") is False
    assert cell_done(str(csv), "B", "CCO") is False
    assert cell_done(str(tmp_path / "missing.csv"), "A", "CCO") is False


def test_discrimination_auroc_direction():
    # knowns are stronger: lower affinity_pred, higher binder_prob than randoms
    df = pd.DataFrame([
        {"class": "known", "affinity_pred": -2.0, "binder_prob": 0.9},
        {"class": "known", "affinity_pred": -1.5, "binder_prob": 0.8},
        {"class": "random", "affinity_pred": 1.0, "binder_prob": 0.2},
        {"class": "random", "affinity_pred": 0.5, "binder_prob": 0.3},
    ])
    # affinity: lower is better -> perfect separation -> AUROC 1.0
    assert discrimination_auroc(df, "affinity_pred", higher_is_better=False) == 1.0
    # binder_prob: higher is better -> perfect separation -> AUROC 1.0
    assert discrimination_auroc(df, "binder_prob", higher_is_better=True) == 1.0


def test_discrimination_auroc_none_when_single_class():
    df = pd.DataFrame([
        {"class": "known", "affinity_pred": -2.0, "binder_prob": 0.9},
        {"class": "known", "affinity_pred": -1.5, "binder_prob": 0.8},
    ])
    assert discrimination_auroc(df, "affinity_pred", higher_is_better=False) is None


def test_discrimination_auroc_ignores_nan():
    df = pd.DataFrame([
        {"class": "known", "affinity_pred": -2.0, "binder_prob": 0.9},
        {"class": "known", "affinity_pred": np.nan, "binder_prob": 0.8},
        {"class": "random", "affinity_pred": 1.0, "binder_prob": 0.2},
    ])
    # one known dropped for NaN; still 1 known vs 1 random, perfect -> 1.0
    assert discrimination_auroc(df, "affinity_pred", higher_is_better=False) == 1.0
