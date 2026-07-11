"""Unit tests for the shard-aware own-pocket phase and --skip-af option added to
scripts/powered_run.py for the crystal-only, sampled-mismatch scaled run (N=41).

No network or docking calls are made here: _select_sources is a pure helper, and the CLI
checks only exercise --help (option wiring), not main()'s body.
"""
from __future__ import annotations

import sys
import pathlib

from click.testing import CliRunner

# Ensure project root is on the path so we can import scripts.powered_run
_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.powered_run import main, _sample_mismatch, _select_sources


def _targets(n):
    return [{"target_id": f"T{i}"} for i in range(n)]


def test_select_sources_defaults_to_all_when_unsharded():
    ok = _targets(5)
    assert _select_sources(ok, None, None) == ok


def test_select_sources_explicit_sources_opt_filters():
    ok = _targets(5)
    sources = _select_sources(ok, "T1,T3", None)
    assert [t["target_id"] for t in sources] == ["T1", "T3"]


def test_select_sources_shard_matches_index_mod_n():
    ok = _targets(9)
    shard0 = _select_sources(ok, None, "0/3")
    assert [t["target_id"] for t in shard0] == ["T0", "T3", "T6"]


def test_select_sources_sources_opt_overrides_source_shard():
    ok = _targets(5)
    sources = _select_sources(ok, "T4", "0/2")
    assert [t["target_id"] for t in sources] == ["T4"]


def test_select_sources_shards_partition_ok_disjointly_and_cover_all():
    ok = _targets(41)
    n = 4
    shards = [_select_sources(ok, None, f"{i}/{n}") for i in range(n)]

    # disjoint: no target_id appears in more than one shard
    seen = set()
    for shard in shards:
        ids = {t["target_id"] for t in shard}
        assert not (ids & seen), "shards must not overlap"
        seen |= ids

    # cover: union of shards == all of ok
    assert seen == {t["target_id"] for t in ok}

    # every target's own shard is the one matching its index%n (sanity on the partition rule)
    for i, t in enumerate(ok):
        assert t in shards[i % n]


def test_select_sources_shard_partition_is_stable_for_various_n():
    ok = _targets(41)
    for n in (1, 2, 3, 5, 7):
        shards = [_select_sources(ok, None, f"{i}/{n}") for i in range(n)]
        total = sum(len(s) for s in shards)
        assert total == len(ok)
        seen = set()
        for shard in shards:
            ids = {t["target_id"] for t in shard}
            assert not (ids & seen)
            seen |= ids
        assert seen == {t["target_id"] for t in ok}


def test_cli_has_skip_af_flag_and_is_off_by_default():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--skip-af" in result.output

    skip_af_param = next(p for p in main.params if p.name == "skip_af")
    assert skip_af_param.is_flag is True
    assert skip_af_param.default is False


def test_sample_mismatch_still_importable_and_passing():
    # Guard against the refactor accidentally breaking the existing --mismatch-sample helper.
    ok = [f"T{i}" for i in range(20)]
    s = _sample_mismatch("T3", ok, k=5, seed=42)
    assert s[0] == "T3"
    assert len(set(s)) == 6
