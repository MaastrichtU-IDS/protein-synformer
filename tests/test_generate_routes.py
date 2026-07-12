"""Pure tests for scripts/generate_routes.py — no torch/model imports at module
scope so these stay cheap in .venv. Run with:
    .venv/bin/python -m pytest tests/test_generate_routes.py -q
"""
from scripts.generate_routes import dedup_keep_first


def test_dedup_keep_first_keeps_first_occurrence_and_order():
    records = [
        {"smiles": "CCO", "v": 1},
        {"smiles": "CCN", "v": 2},
        {"smiles": "CCO", "v": 3},  # dup, later — must be dropped
        {"smiles": "c1ccccc1", "v": 4},
        {"smiles": "CCN", "v": 5},  # dup, later — must be dropped
    ]
    out = dedup_keep_first(records)
    assert [r["smiles"] for r in out] == ["CCO", "CCN", "c1ccccc1"]
    assert [r["v"] for r in out] == [1, 2, 4]  # first occurrence's payload kept


def test_dedup_keep_first_empty():
    assert dedup_keep_first([]) == []


def test_dedup_keep_first_no_dupes_is_identity_order():
    records = [{"smiles": "A"}, {"smiles": "B"}, {"smiles": "C"}]
    assert dedup_keep_first(records) == records


class _FakeResult:
    """Minimal stand-in for GenerateResult: just the 4 route tensors
    routes_from_result reads, at a given batch size N."""

    def __init__(self, n, seq_len=5):
        import torch

        self.token_types = torch.randint(0, 4, (n, seq_len))
        self.rxn_indices = torch.randint(0, 10, (n, seq_len))
        self.reactant_fps = torch.rand(n, seq_len, 8)
        self.token_padding_mask = torch.zeros(n, seq_len, dtype=torch.bool)


def test_routes_from_result_shape_matches_batch_size():
    from synformer.molopt.dpo import routes_from_result

    n = 4
    result = _FakeResult(n)
    routes = routes_from_result(result)
    assert len(routes) == n
    for i, r in enumerate(routes):
        for key in ("token_types", "rxn_indices", "reactant_fps", "token_padding_mask"):
            assert key in r
            assert r[key].size(0) == 1  # batch-dim-1 slice
        # content matches row i of the original batch
        assert (r["token_types"][0] == result.token_types[i]).all()
        assert (r["rxn_indices"][0] == result.rxn_indices[i]).all()
