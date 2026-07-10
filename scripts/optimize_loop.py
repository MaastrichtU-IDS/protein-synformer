"""Frozen-model enrichment-selection loop orchestrator (runs in .venv).
Generation is delegated to .venv-train via subprocess (Task 7)."""
from __future__ import annotations

import json
import math
import pathlib
from concurrent.futures import ThreadPoolExecutor

from synformer.molopt.enrich import (
    EnrichWeights, compute_enrichment_weights, molecule_index_sets, passes_gate,
)


def read_candidates(path: str | pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def gate_and_dedup(records: list[dict], sa_max: float = 4.0) -> list[dict]:
    out, seen = [], set()
    for r in records:
        smi = r["smiles"]
        if smi in seen:
            continue
        if passes_gate(smi, sa_max=sa_max):
            seen.add(smi)
            out.append(r)
    return out


def dock_budget(records, spec, dock_fn, budget, seed, max_workers=4) -> dict[str, float]:
    picks = records[:budget]

    def _one(r):
        return r["smiles"], dock_fn(spec, r["smiles"], seed=seed)

    scored: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for smi, score in ex.map(_one, picks):
            if score is not None and not math.isnan(score):
                scored[smi] = float(score)
    return scored


def select_winners(scored: dict[str, float], k: int) -> list[str]:
    return [s for s, _ in sorted(scored.items(), key=lambda kv: kv[1])[:k]]


def next_weights(winner_smiles, all_records, w_max: float = 5.0) -> dict:
    by_smi = {r["smiles"]: r for r in all_records}
    def sets(smi):
        r = by_smi[smi]
        return molecule_index_sets(r["bb"], r["tpl"])
    winners = [sets(s) for s in winner_smiles if s in by_smi]
    pool = [molecule_index_sets(r["bb"], r["tpl"]) for r in all_records]
    ew: EnrichWeights = compute_enrichment_weights(winners, pool, w_max=w_max)
    return {"bb": {str(k): v for k, v in ew.bb.items()},
            "tpl": {str(k): v for k, v in ew.tpl.items()}}
