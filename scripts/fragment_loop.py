"""Fragment-seeding hill-climb: dock -> top-k seeds -> analog-sample -> dock -> re-seed.
Three budget-matched arms (treatment / control_a / control_b). Runs in .venv;
analog + pocket generation are delegated to .venv-train subprocesses (Task 4)."""
from __future__ import annotations

import random

from scripts.optimize_loop import (  # reused, do not duplicate
    dock_budget, gate_and_dedup, read_candidates, select_winners,
)


def select_topk_seeds(scored: dict[str, float], k: int) -> list[str]:
    return select_winners(scored, k)


def select_random_seeds(scored: dict[str, float], k: int, seed: int) -> list[str]:
    pool = [s for s, v in scored.items() if v == v]  # drop nan
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:k]
