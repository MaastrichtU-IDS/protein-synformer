import numpy as np
import pandas as pd
import pytest

from scripts.dpo_eval import own_preference, two_sample_diff_ci, joint_z_own_preference


def _build_df():
    # Held-out eval frame: molecules docked into the held-out target pocket (T0) plus two
    # mismatch pockets (T1, T2). Lower score = better binding (smina, kcal/mol).
    rows = [
        # A: binds its OWN pocket (T0) far better than either mismatch pocket -> should
        # strongly PREFER its own pocket, i.e. d(A) > 0 and large.
        ("T0", "T0", "A", "candidate", -10.0),
        ("T0", "T1", "A", "candidate", -4.0),
        ("T0", "T2", "A", "candidate", -6.0),
        # C: binds every pocket about equally well -> promiscuous, d(C) ~= 0.
        ("T0", "T0", "C", "candidate", -7.0),
        ("T0", "T1", "C", "candidate", -7.0),
        ("T0", "T2", "C", "candidate", -7.0),
        # D: mismatch pockets docked, but NEVER docked into its own pocket (T0) -> must be
        # skipped (no own score).
        ("T0", "T1", "D", "candidate", -9.0),
        ("T0", "T2", "D", "candidate", -9.0),
        # E: docked into its own pocket only, no mismatch pockets at all -> must be
        # skipped (no mismatch scores).
        ("T0", "T0", "E", "candidate", -9.0),
        # F: a "known"/reference row (not source=="candidate") that must never leak in,
        # even though it has both own and mismatch cells.
        ("T0", "T0", "F", "known", -20.0),
        ("T0", "T1", "F", "known", -1.0),
    ]
    return pd.DataFrame(rows, columns=["target", "pocket", "molecule", "source", "score"])


def test_own_preference_hand_computed():
    df = _build_df()
    d = own_preference(df, target="T0", mismatch_pockets=["T1", "T2"])

    # D (no own score) and E (no mismatch scores) are skipped; F is filtered by source.
    assert set(d.keys()) == {"A", "C"}

    # A: own_score = -10.0 (best/min at T0). mismatch_scores = [-4.0, -6.0] (best per
    # pocket) -> mean(mismatch) = -5.0. d(A) = mean(mismatch) - own = -5.0 - (-10.0) = 5.0.
    assert d["A"] == pytest.approx(5.0)
    # C: own_score = -7.0, mismatch_scores = [-7.0, -7.0] -> mean = -7.0.
    # d(C) = -7.0 - (-7.0) = 0.0.
    assert d["C"] == pytest.approx(0.0)

    # SIGN CONVENTION: d > 0 means the molecule prefers its own pocket. The specific
    # molecule A must score higher (more own-preferring) than the promiscuous molecule C.
    assert d["A"] > 0
    assert d["C"] == pytest.approx(0.0)
    assert d["A"] > d["C"]


def test_own_preference_duplicate_dock_takes_best_min_score():
    # Molecule A docked twice into its own pocket; the WORSE (less negative) duplicate
    # must be ignored in favor of the best (min) score.
    rows = [
        ("T0", "T0", "A", "candidate", -10.0),
        ("T0", "T0", "A", "candidate", -2.0),  # worse duplicate, must be dropped
        ("T0", "T1", "A", "candidate", -5.0),
    ]
    df = pd.DataFrame(rows, columns=["target", "pocket", "molecule", "source", "score"])
    d = own_preference(df, target="T0", mismatch_pockets=["T1"])
    # own_score should be -10.0 (min), not -2.0 -> d = -5.0 - (-10.0) = 5.0.
    assert d["A"] == pytest.approx(5.0)


def test_own_preference_smiles_subset_restricts():
    df = _build_df()
    d = own_preference(df, target="T0", mismatch_pockets=["T1", "T2"], smiles_subset={"A"})
    assert set(d.keys()) == {"A"}
    assert d["A"] == pytest.approx(5.0)


def test_own_preference_empty_when_no_candidates_match():
    df = _build_df()
    d = own_preference(df, target="T0", mismatch_pockets=["T1", "T2"], smiles_subset={"NOPE"})
    assert d == {}


def test_own_preference_nan_scores_ignored():
    rows = [
        ("T0", "T0", "A", "candidate", -10.0),
        ("T0", "T1", "A", "candidate", float("nan")),
        ("T0", "T2", "A", "candidate", -5.0),
    ]
    df = pd.DataFrame(rows, columns=["target", "pocket", "molecule", "source", "score"])
    d = own_preference(df, target="T0", mismatch_pockets=["T1", "T2"])
    # T1 score is NaN so only T2's -5.0 counts as the mismatch score.
    assert d["A"] == pytest.approx(-5.0 - (-10.0))


# --- two_sample_diff_ci -------------------------------------------------------------

def test_two_sample_diff_ci_detects_positive_shift():
    rng = np.random.default_rng(0)
    a = list(rng.normal(loc=5.0, scale=0.5, size=200))
    b = list(rng.normal(loc=1.0, scale=0.5, size=200))
    diff, lo, hi = two_sample_diff_ci(a, b, seed=42, n_boot=2000)
    assert diff > 0
    assert lo > 0  # CI excludes 0


def test_two_sample_diff_ci_identical_distributions_includes_zero():
    rng = np.random.default_rng(1)
    vals = list(rng.normal(loc=0.0, scale=1.0, size=200))
    diff, lo, hi = two_sample_diff_ci(vals, vals, seed=42, n_boot=2000)
    assert diff == pytest.approx(0.0)
    assert lo < 0 < hi


def test_two_sample_diff_ci_deterministic_same_seed():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [0.5, 1.5, 2.5, 3.5, 4.5]
    r1 = two_sample_diff_ci(a, b, seed=7, n_boot=500)
    r2 = two_sample_diff_ci(a, b, seed=7, n_boot=500)
    assert r1 == r2


def test_two_sample_diff_ci_observed_diff_is_plain_mean_difference():
    a = [1.0, 2.0, 3.0]
    b = [10.0, 20.0, 30.0]
    diff, _, _ = two_sample_diff_ci(a, b, seed=1, n_boot=100)
    assert diff == pytest.approx(np.mean(a) - np.mean(b))


# --- joint_z_own_preference ----------------------------------------------------------

def _build_joint_df():
    # Pockets: T0 = held-out target, T1/T2 = mismatch (train) pockets.
    data = {
        "MB1": [-10.0, -5.0, -5.0],   # base, clearly specific
        "MB2": [-6.0, -6.0, -6.0],    # base, promiscuous
        "MD1": [-9.0, -5.0, -6.0],    # dpo, specific
        "MD2": [-6.5, -6.0, -7.0],    # dpo, promiscuous
    }
    rows = []
    for mol, (t0, t1, t2) in data.items():
        rows.append(("T0", "T0", mol, "candidate", t0))
        rows.append(("T0", "T1", mol, "candidate", t1))
        rows.append(("T0", "T2", mol, "candidate", t2))
    return pd.DataFrame(rows, columns=["target", "pocket", "molecule", "source", "score"])


def test_joint_z_own_preference_splits_by_origin_and_sign():
    df = _build_joint_df()
    origin = {"MB1": "base", "MB2": "base", "MD1": "dpo", "MD2": "dpo"}
    out = joint_z_own_preference(df, target="T0", mismatch_pockets=["T1", "T2"], origin_by_smiles=origin)

    assert set(out.keys()) == {"base", "dpo"}
    assert set(out["base"].keys()) == {"MB1", "MB2"}
    assert set(out["dpo"].keys()) == {"MD1", "MD2"}

    # SIGN CONVENTION (opposite of own_preference's raw d): more NEGATIVE dz = more
    # specific, matching scripts/dpo_pairs.py::per_molecule_specificity /
    # scripts/powered_analyze.py::_delta_win_from_matrix. MB1 is clearly own-pocket
    # specific (own score far better than both mismatches) so its dz must be negative.
    assert out["base"]["MB1"] < 0

    # Hand-derived (cross-checked with a standalone numpy computation over the joint
    # 4-molecule x 3-pocket matrix, nan-aware column z-normalization, axis=0):
    #   column T0: [-10,-6,-9,-6.5]  -> mean -7.875, std 1.67238602...
    #   column T1: [-5,-6,-5,-6]     -> mean -5.5,   std 0.5
    #   column T2: [-5,-6,-6,-7]     -> mean -6.0,   std 0.70710678...
    assert out["base"]["MB1"] == pytest.approx(-2.4777464388648385, abs=1e-9)
    assert out["base"]["MB2"] == pytest.approx(1.6211526391279039, abs=1e-9)
    assert out["dpo"]["MD1"] == pytest.approx(-1.1726915834767424, abs=1e-9)
    assert out["dpo"]["MD2"] == pytest.approx(2.029285383213677, abs=1e-9)


def test_joint_z_own_preference_missing_own_or_mismatch_skipped():
    rows = [
        ("T0", "T0", "MB1", "candidate", -10.0),
        ("T0", "T1", "MB1", "candidate", -5.0),
        ("T0", "T2", "MB1", "candidate", -5.0),
        # MB2: a second, fully-populated molecule so the per-column std used for
        # z-normalization is non-degenerate (nonzero) once NOMISS/NOMISM drop out.
        ("T0", "T0", "MB2", "candidate", -6.0),
        ("T0", "T1", "MB2", "candidate", -6.0),
        ("T0", "T2", "MB2", "candidate", -6.0),
        # NOMISS: no own-pocket (T0) row at all -> must be skipped.
        ("T0", "T1", "NOMISS", "candidate", -9.0),
        ("T0", "T2", "NOMISS", "candidate", -9.0),
        # NOMISM: own-pocket row only, no mismatch pockets at all -> must be skipped.
        ("T0", "T0", "NOMISM", "candidate", -9.0),
    ]
    df = pd.DataFrame(rows, columns=["target", "pocket", "molecule", "source", "score"])
    origin = {"MB1": "base", "MB2": "base", "NOMISS": "base", "NOMISM": "base"}
    out = joint_z_own_preference(df, target="T0", mismatch_pockets=["T1", "T2"], origin_by_smiles=origin)
    assert set(out["base"].keys()) == {"MB1", "MB2"}


def test_joint_z_own_preference_filters_to_candidate_source_only():
    rows = [
        ("T0", "T0", "MB1", "candidate", -10.0),
        ("T0", "T1", "MB1", "candidate", -5.0),
        ("T0", "T2", "MB1", "candidate", -5.0),
        # MB2: a second candidate molecule so the column std is non-degenerate once the
        # "known" reference rows below are filtered out.
        ("T0", "T0", "MB2", "candidate", -6.0),
        ("T0", "T1", "MB2", "candidate", -6.0),
        ("T0", "T2", "MB2", "candidate", -6.0),
        ("T0", "T0", "KNOWN1", "known", -20.0),
        ("T0", "T1", "KNOWN1", "known", -1.0),
        ("T0", "T2", "KNOWN1", "known", -1.0),
    ]
    df = pd.DataFrame(rows, columns=["target", "pocket", "molecule", "source", "score"])
    origin = {"MB1": "base", "MB2": "base", "KNOWN1": "base"}
    out = joint_z_own_preference(df, target="T0", mismatch_pockets=["T1", "T2"], origin_by_smiles=origin)
    assert "KNOWN1" not in out["base"]
    assert "MB1" in out["base"]
    assert "MB2" in out["base"]
