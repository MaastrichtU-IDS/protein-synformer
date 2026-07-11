import os
import subprocess
import sys

from scripts.powered_run import _sample_mismatch


def test_sample_includes_own_first_and_k_distinct_others():
    ok = [f"T{i}" for i in range(20)]
    s = _sample_mismatch("T3", ok, k=5, seed=42)
    assert s[0] == "T3"                      # own first
    assert "T3" not in s[1:]                 # own not duplicated
    assert len(s) == 6                       # own + 5
    assert len(set(s)) == 6                  # distinct
    assert all(x in ok for x in s)


def test_sample_is_seeded_deterministic():
    ok = [f"T{i}" for i in range(20)]
    assert _sample_mismatch("T3", ok, 5, 42) == _sample_mismatch("T3", ok, 5, 42)


def test_sample_k_ge_pool_returns_all():
    ok = ["A", "B", "C"]
    s = _sample_mismatch("A", ok, k=10, seed=1)
    assert s[0] == "A" and set(s) == {"A", "B", "C"} and len(s) == 3


def _run(hashseed):
    code = ("from scripts.powered_run import _sample_mismatch;"
            "print(_sample_mismatch('T3', [f'T{i}' for i in range(20)], 5, 42))")
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
    ).stdout


def test_sample_reproducible_across_process_hashseed():
    # tuple/string-hash seeding is salted by PYTHONHASHSEED and would differ across
    # processes; a deterministic string seed must produce identical output regardless.
    assert _run("0") == _run("1")
