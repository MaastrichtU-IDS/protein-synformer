"""Unit tests for the idempotency helper in scripts/powered_run.py.

No network or docking calls are made here.
"""
from __future__ import annotations

import csv
import sys
import pathlib

# Ensure project root is on the path so we can import scripts.powered_run
_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.powered_run import _done_pairs


def test_done_pairs_reads_molecule_pocket_tuples(tmp_path):
    csv_path = tmp_path / "scores.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "pocket", "molecule", "source", "score"])
        w.writeheader()
        w.writerow({"target": "A_WT", "pocket": "A_WT", "molecule": "CCO", "source": "candidate", "score": -5.0})
        w.writerow({"target": "A_WT", "pocket": "B_WT", "molecule": "CCO", "source": "candidate", "score": -4.0})

    done = _done_pairs(str(csv_path))
    assert ("CCO", "A_WT") in done
    assert ("CCO", "B_WT") in done
    assert ("CCN", "A_WT") not in done  # absent molecule/pocket pair


def test_done_pairs_missing_file_returns_empty_set(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    assert _done_pairs(str(missing)) == set()
