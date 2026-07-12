import pandas as pd
import pytest

from scripts.dpo_pairs import make_pairs, per_molecule_specificity


def _build_df():
    # Docking frame for source target T0, docked into own pocket T0 and mismatch pockets
    # T1/T2. Lower score = better binding (smina, kcal/mol).
    rows = [
        # A: binds its OWN pocket (T0) far better than either mismatch pocket -> SPECIFIC.
        ("T0", "T0", "A", "candidate", -10.0),
        ("T0", "T1", "A", "candidate", -5.0),
        ("T0", "T2", "A", "candidate", -5.0),
        # B: decent own-pocket score, only ONE mismatch pocket docked (T1 row missing) ->
        # must still be scored from the remaining (T2) mismatch cell.
        ("T0", "T0", "B", "candidate", -8.0),
        ("T0", "T2", "B", "candidate", -6.0),
        # C: binds every pocket about equally well -> PROMISCUOUS.
        ("T0", "T0", "C", "candidate", -7.0),
        ("T0", "T1", "C", "candidate", -7.0),
        ("T0", "T2", "C", "candidate", -7.0),
        # D: mismatch pockets docked, but NEVER docked into its own pocket -> must be
        # skipped entirely (no own cell to compute z_own from).
        ("T0", "T1", "D", "candidate", -9.0),
        ("T0", "T2", "D", "candidate", -9.0),
        # E: docked into its own pocket only, no mismatch pockets at all -> must be
        # skipped entirely (no mismatch cell to average).
        ("T0", "T0", "E", "candidate", -9.0),
    ]
    return pd.DataFrame(rows, columns=["target", "pocket", "molecule", "source", "score"])


def test_per_molecule_specificity_hand_computed():
    df = _build_df()
    spec = per_molecule_specificity(df, "T0")

    # D (no own cell) and E (no mismatch cells) are skipped entirely.
    assert set(spec.keys()) == {"A", "B", "C"}

    # Exact numbers, hand-derived from the per-column z-normalization described in the
    # docstring and cross-checked against a standalone numpy computation (see
    # task-3-report.md for the full derivation):
    #   column T0 (own): values [-10,-8,-7,-9] (A,B,C,E) -> mean -8.5, std sqrt(1.25)
    #   column T1: values [-5,-7,-9] (A,C,D)             -> mean -7.0, std sqrt(8/3)
    #   column T2: values [-5,-6,-7,-9] (A,B,C,D)        -> mean -6.75, std sqrt(2.1875)
    assert spec["A"] == pytest.approx(-2.5456212005056296, abs=1e-9)
    assert spec["B"] == pytest.approx(-0.05987895733715204, abs=1e-9)
    assert spec["C"] == pytest.approx(1.4261562119727256, abs=1e-9)

    # SIGN CONVENTION: the specific molecule (A) must be more negative than the
    # promiscuous one (C). Getting this backwards trains DPO in the wrong direction.
    assert spec["A"] < spec["C"]
    assert spec["A"] < 0
    assert spec["C"] > 0


def test_per_molecule_specificity_duplicate_dock_takes_min_best_score():
    # Molecule A is docked twice into its own pocket; the WORSE (less negative, i.e.
    # numerically larger) duplicate must be ignored and the best (min) score used.
    rows = [
        ("T0", "T0", "A", "candidate", -10.0),
        ("T0", "T0", "A", "candidate", -3.0),  # worse duplicate -> must be dropped
        ("T0", "T1", "A", "candidate", -5.0),
        ("T0", "T0", "C", "candidate", -7.0),
        ("T0", "T1", "C", "candidate", -7.0),
    ]
    df = pd.DataFrame(rows, columns=["target", "pocket", "molecule", "source", "score"])
    spec = per_molecule_specificity(df, "T0")
    assert spec["A"] < spec["C"]


def test_per_molecule_specificity_no_rows_for_target_returns_empty():
    df = _build_df()
    assert per_molecule_specificity(df, "NOPE") == {}


def test_per_molecule_specificity_filters_to_requested_target_only():
    # A second source target (T5) with its own molecules/pockets must not leak into T0's
    # specificity computation (and vice versa).
    df = _build_df()
    other = pd.DataFrame(
        [("T5", "T5", "X", "candidate", -1.0), ("T5", "T6", "X", "candidate", -1.0)],
        columns=["target", "pocket", "molecule", "source", "score"],
    )
    combined = pd.concat([df, other], ignore_index=True)
    spec = per_molecule_specificity(combined, "T0")
    assert set(spec.keys()) == {"A", "B", "C"}
    assert spec["A"] == pytest.approx(-2.5456212005056296, abs=1e-9)


def test_make_pairs_direction_and_composition():
    spec = {
        "most_specific": -3.0,
        "specific2": -2.0,
        "specific3": -1.5,
        "mid1": 0.0,
        "mid2": 0.05,
        "mid3": 0.1,
        "mid4": 0.2,
        "promisc3": 1.5,
        "promisc2": 2.0,
        "most_promiscuous": 3.0,
    }  # n=10, frac=0.3 -> k = 3
    pairs = make_pairs(spec, frac=0.3)

    assert len(pairs) > 0
    winners = {w for w, _ in pairs}
    losers = {l for _, l in pairs}

    # SIGN CONVENTION: the most-specific (most negative) molecule must appear ONLY as a
    # winner, and the most-promiscuous (most positive) ONLY as a loser.
    assert "most_specific" in winners
    assert "most_specific" not in losers
    assert "most_promiscuous" in losers
    assert "most_promiscuous" not in winners

    assert winners == {"most_specific", "specific2", "specific3"}
    assert losers == {"promisc3", "promisc2", "most_promiscuous"}

    for w, l in pairs:
        assert w != l
        assert spec[w] < spec[l]


def test_make_pairs_pair_count_is_full_cross_product():
    spec = {f"w{i}": -float(i + 1) for i in range(3)}
    spec.update({f"l{i}": float(i + 1) for i in range(3)})
    pairs = make_pairs(spec, frac=0.5)  # n=6, k=3
    assert len(pairs) == 3 * 3
    assert {w for w, _ in pairs} == {"w0", "w1", "w2"}
    assert {l for _, l in pairs} == {"l0", "l1", "l2"}


def test_make_pairs_too_few_molecules_returns_empty():
    assert make_pairs({}, frac=0.3) == []
    assert make_pairs({"a": -1.0}, frac=0.3) == []
    assert make_pairs({"a": -1.0, "b": 1.0}, frac=0.3) == []  # floor(2*0.3)=0
