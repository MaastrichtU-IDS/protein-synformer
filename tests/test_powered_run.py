"""Unit tests for the panel selector in scripts/powered_run.py.

No network or docking calls are made here.
"""
from __future__ import annotations

import sys
import pathlib

# Ensure project root is on the path so we can import scripts.powered_run
_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.powered_run import choose_panel


def test_choose_panel_is_seeded_and_distinct():
    ts = [f"T{i}_WT" for i in range(20)]
    a = choose_panel(ts, p=6, seed=42)
    b = choose_panel(ts, p=6, seed=42)
    assert a == b                      # reproducible
    assert len(a) == 6 and len(set(a)) == 6   # distinct
    assert set(a).issubset(set(ts))
    assert choose_panel(ts, p=6, seed=7) != a  # seed changes it


def test_choose_panel_caps_at_available():
    ts = ["A_WT", "B_WT", "C_WT"]
    assert set(choose_panel(ts, p=6, seed=1)) == set(ts)  # can't exceed N
