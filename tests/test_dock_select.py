"""Unit tests for pure helpers in scripts/dock_select.py.

No network or docking calls are made here.
"""
from __future__ import annotations

import math
import sys
import pathlib

# Ensure project root is on the path so we can import scripts.dock_select
_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.dock_select import select_topm_for_target, _safe_mean


# ── select_topm_for_target ─────────────────────────────────────────────────────

class TestSelectTopmForTarget:
    def test_basic_selection(self):
        """Returns the M lowest-score SMILES."""
        scores = {"A": -7.0, "B": -5.0, "C": -8.0, "D": -6.0}
        result = select_topm_for_target(scores, m=2)
        # Best (lowest) are C (-8.0) then A (-7.0)
        assert result == ["C", "A"]

    def test_excludes_nan(self):
        """NaN scores are excluded from selection."""
        scores = {"A": float("nan"), "B": -5.0, "C": -8.0}
        result = select_topm_for_target(scores, m=3)
        assert "A" not in result
        assert result == ["C", "B"]

    def test_m_larger_than_finite(self):
        """When M > finite entries, returns all finite entries."""
        scores = {"A": -7.0, "B": float("nan")}
        result = select_topm_for_target(scores, m=5)
        assert result == ["A"]

    def test_all_nan(self):
        """All NaN → empty list."""
        scores = {"A": float("nan"), "B": float("nan")}
        result = select_topm_for_target(scores, m=2)
        assert result == []

    def test_empty_input(self):
        """Empty dict → empty list."""
        result = select_topm_for_target({}, m=3)
        assert result == []

    def test_restart_union_case(self):
        """Simulates restart: some scores come from disk, some are fresh.

        The top-M must be selected from the FULL union, not just the fresh subset.
        The best molecule is 'disk_best' which was already scored on a prior run.
        """
        # Scores "from disk" (already in scores_table before docking loop)
        disk_scores = {
            "disk_best": -9.5,   # best overall — would be MISSED without union
            "disk_mid": -6.0,
        }
        # Scores "freshly docked" in this run
        fresh_scores = {
            "fresh_a": -7.0,
            "fresh_b": -5.0,
            "fresh_c": -8.0,
        }
        # Union (as produced by the D1 fix: build own_pocket_scores from scores_table)
        all_scores = {**disk_scores, **fresh_scores}

        result = select_topm_for_target(all_scores, m=2)
        # Top-2: disk_best (-9.5), fresh_c (-8.0)
        assert result[0] == "disk_best", "Best on-disk molecule must be selected"
        assert result[1] == "fresh_c"
        assert len(result) == 2

    def test_restart_union_matches_fresh_full(self):
        """Restart + union must give the SAME top-M as a single fresh run over all scores."""
        all_scores = {"A": -9.0, "B": -7.0, "C": -8.5, "D": -5.0, "E": -6.0}

        # Simulate fresh run (all scored in one go)
        fresh_result = select_topm_for_target(all_scores, m=3)

        # Simulate restart: half were on disk, half are fresh — union is identical
        disk_scores = {"A": -9.0, "B": -7.0}
        new_scores = {"C": -8.5, "D": -5.0, "E": -6.0}
        restart_result = select_topm_for_target({**disk_scores, **new_scores}, m=3)

        assert fresh_result == restart_result


# ── _safe_mean ─────────────────────────────────────────────────────────────────

class TestSafeMean:
    def test_basic(self):
        assert math.isclose(_safe_mean([-2.0, -4.0, -6.0]), -4.0)

    def test_empty_returns_nan(self):
        assert math.isnan(_safe_mean([]))

    def test_all_nan_returns_nan(self):
        assert math.isnan(_safe_mean([float("nan"), float("nan")]))

    def test_mixed_nan(self):
        """NaN entries are ignored; only finite values contribute."""
        result = _safe_mean([-3.0, float("nan"), -5.0])
        assert math.isclose(result, -4.0)

    def test_single_value(self):
        assert math.isclose(_safe_mean([-7.5]), -7.5)
