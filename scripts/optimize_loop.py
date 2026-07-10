"""Frozen-model enrichment-selection loop orchestrator (runs in .venv).
Generation is delegated to .venv-train via subprocess (Task 7)."""
from __future__ import annotations

import csv
import json
import math
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

import click
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from synformer.dock.dock import dock
from synformer.dock.receptor import prepare_target
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


def round_dir(base, target, arm, r) -> pathlib.Path:
    return pathlib.Path(base) / target / arm / f"round_{r}"


def is_round_done(rd: pathlib.Path) -> bool:
    p = pathlib.Path(rd) / "dock_scores.csv"
    return p.exists() and p.stat().st_size > 0 and len(p.read_text().splitlines()) > 1


def run_generation(ckpt, target, weights_path, n, seed, out_path,
                    python=".venv-train/bin/python"):
    cmd = [python, "-m", "scripts.generate_enriched", "--ckpt", ckpt, "--target", target,
           "--weights", str(weights_path), "--n", str(n), "--seed", str(seed), "--out", str(out_path)]
    subprocess.run(cmd, check=True)


def _scaffold_diversity(smiles_list) -> float:
    scaffs = set()
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            scaffs.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
    return len(scaffs) / max(1, len(smiles_list))


def run_arm(ckpt, target, arm, spec, rounds, budget, n, k, seed, out_dir,
            summary_rows=None) -> list[str]:
    all_scores: dict[str, float] = {}
    weights_path = "NONE"
    for r in range(rounds):
        rd = round_dir(out_dir, target, arm, r)
        rd.mkdir(parents=True, exist_ok=True)
        cand = rd / "candidates.jsonl"
        scores_csv = rd / "dock_scores.csv"
        if is_round_done(rd):
            # gate on resume too, so the enrichment pool denominator matches the fresh path
            recs = gate_and_dedup(read_candidates(cand))
            with open(scores_csv, newline="") as fh:
                scored = {row["smiles"]: float(row["score"]) for row in csv.DictReader(fh)}
        else:
            run_generation(ckpt, target, weights_path if arm == "enrich" else "NONE",
                            n, seed + r, cand)
            recs = gate_and_dedup(read_candidates(cand))
            rseed = seed + r
            scored = dock_budget(recs, spec, dock, budget, rseed)
            with open(scores_csv, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["smiles", "score"])
                for s, v in scored.items():
                    w.writerow([s, v])
        all_scores.update(scored)
        winners = select_winners(scored, k)
        if arm == "enrich":
            # enrichment denominator is the DOCKED pool, not the full gated pool
            docked_recs = [r for r in recs if r["smiles"] in scored]
            nw = next_weights(winners, docked_recs)
            wp = rd / "weights_next.json"; wp.write_text(json.dumps(nw))
            weights_path = str(wp)
        if summary_rows is not None:
            top10 = sorted(scored.values())[:10]
            summary_rows.append({
                "target": target, "arm": arm, "round": r,
                "n_gated": len(recs), "n_docked": len(scored),
                "best": min(scored.values()) if scored else float("nan"),
                "top10_mean": sum(top10) / len(top10) if top10 else float("nan"),
                "scaffold_div": _scaffold_diversity(list(scored)),
            })
    return select_winners(all_scores, k)


@click.command()
@click.option("--targets", default="data/dock/powered_targets.json")
@click.option("--ckpt", required=True)
@click.option("--arms", default="enrich,uniform")
@click.option("--rounds", default=3, type=int)
@click.option("--budget", default=150, type=int)
@click.option("--n", default=1000, type=int)
@click.option("--k", default=30, type=int)
@click.option("--final-m", default=10, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--out-dir", default="data/dock/sp_l")
@click.option("--limit-targets", default=None, type=int)
@click.option("--work-dir", default="boltz_out/sp_l")
def main(targets, ckpt, arms, rounds, budget, n, k, final_m, seed, out_dir, limit_targets, work_dir):
    import os
    tgts = json.load(open(targets))
    if limit_targets:
        tgts = tgts[:limit_targets]
    arm_list = [a.strip() for a in arms.split(",")]
    summary_rows: list[dict] = []
    for t in tgts:
        tid = t["target_id"]
        spec = prepare_target(t["pdb_id"], f"{work_dir}/holo/{tid}", ligand_resname=t["ligand_resname"])
        for arm in arm_list:
            final = run_arm(ckpt, tid, arm, spec, rounds, budget, n, k, seed,
                            out_dir, summary_rows)
            fdir = pathlib.Path(out_dir) / tid / arm
            (fdir / "final_topM.smi").write_text("\n".join(final[:final_m]))
            print(f"  {tid}/{arm}: final top-{final_m} written", flush=True)
    sp = pathlib.Path(out_dir) / "loop_summary.csv"
    os.makedirs(sp.parent, exist_ok=True)
    with open(sp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "arm", "round", "n_gated", "n_docked",
                                           "best", "top10_mean", "scaffold_div"])
        w.writeheader(); w.writerows(summary_rows)
    print(f"loop_summary.csv written ({len(summary_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
