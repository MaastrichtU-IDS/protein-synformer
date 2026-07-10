from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnrichWeights:
    """Sparse per-index enrichment multipliers; a missing index means weight 1.0."""
    bb: dict[int, float] = field(default_factory=dict)
    tpl: dict[int, float] = field(default_factory=dict)


def molecule_index_sets(
    mol_idx_seq: list[int | None], rxn_idx_seq: list[int | None]
) -> tuple[frozenset[int], frozenset[int]]:
    bb = frozenset(i for i in mol_idx_seq if i is not None and i >= 0)
    tpl = frozenset(i for i in rxn_idx_seq if i is not None and i >= 0)
    return bb, tpl


def _weights_for_axis(
    winner_sets: list[frozenset[int]], pool_sets: list[frozenset[int]], w_max: float, eps: float
) -> dict[int, float]:
    n_win = len(winner_sets)
    n_pool = len(pool_sets)
    if n_win == 0 or n_pool == 0:
        return {}
    win_count: dict[int, int] = {}
    for s in winner_sets:
        for i in s:
            win_count[i] = win_count.get(i, 0) + 1
    pool_count: dict[int, int] = {}
    for s in pool_sets:
        for i in s:
            pool_count[i] = pool_count.get(i, 0) + 1
    out: dict[int, float] = {}
    for i, wc in win_count.items():
        f_win = wc / n_win
        f_pool = pool_count.get(i, 0) / n_pool
        ratio = f_win / (f_pool + eps)
        w = max(1.0, min(w_max, ratio))  # only promote (floor 1.0), clip at w_max
        if w > 1.0:
            out[i] = w
    return out


def compute_enrichment_weights(
    winners: list[tuple[frozenset[int], frozenset[int]]],
    pool: list[tuple[frozenset[int], frozenset[int]]],
    w_max: float = 5.0,
    eps: float = 1e-3,
) -> EnrichWeights:
    if not winners or not pool:
        return EnrichWeights()
    bb = _weights_for_axis([w[0] for w in winners], [p[0] for p in pool], w_max, eps)
    tpl = _weights_for_axis([w[1] for w in winners], [p[1] for p in pool], w_max, eps)
    return EnrichWeights(bb=bb, tpl=tpl)
