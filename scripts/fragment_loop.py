"""Fragment-seeding hill-climb: dock -> top-k seeds -> analog-sample -> dock -> re-seed.
Three budget-matched arms (treatment / control_a / control_b). Runs in .venv;
analog + pocket generation are delegated to .venv-train subprocesses (Task 4)."""
from __future__ import annotations

import csv
import json
import pathlib
import random
import subprocess

import click
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from scripts.optimize_loop import (  # reused, do not duplicate
    dock_budget, gate_and_dedup, read_candidates, select_winners,
)
from synformer.dock.dock import dock
from synformer.dock.receptor import prepare_target


def select_topk_seeds(scored: dict[str, float], k: int) -> list[str]:
    return select_winners(scored, k)


def select_random_seeds(scored: dict[str, float], k: int, seed: int) -> list[str]:
    pool = [s for s, v in scored.items() if v == v]  # drop nan
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:k]


def round_dir(base, target, arm, r) -> pathlib.Path:
    return pathlib.Path(base) / target / arm / f"round_{r}"


def is_round_done(rd) -> bool:
    p = pathlib.Path(rd) / "dock_scores.csv"
    return p.exists() and p.stat().st_size > 0 and len(p.read_text().splitlines()) > 1


def run_analog_generation(seeds, model, out, python=".venv-train/bin/python"):
    sp = pathlib.Path(out).with_suffix(".seeds.smi")
    sp.write_text("\n".join(seeds))
    subprocess.run([python, "-m", "scripts.generate_analogs", "--seeds", str(sp),
                    "--model", str(model), "--out", str(out), "--num-gpus", "1"], check=True)


def run_pocket_generation(target, out, ckpt, n, seed, python=".venv-train/bin/python"):
    subprocess.run([python, "-m", "scripts.generate_enriched", "--ckpt", str(ckpt),
                    "--target", target, "--weights", "NONE", "--n", str(n),
                    "--seed", str(seed), "--out", str(out)], check=True)


def _scaffold_diversity(smiles_list) -> float:
    scaffs = set()
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            scaffs.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
    return len(scaffs) / max(1, len(smiles_list))


def run_arm(arm, target, spec, ckpt_analog, ckpt_pocket, rounds, budget, k, n, seed,
            out_dir, round0_scores, summary_rows):
    # round 0 is shared across arms (docked once in main); seed selection reads it
    all_scores = dict(round0_scores)
    for r in range(rounds):
        rd = round_dir(out_dir, target, arm, r)
        rd.mkdir(parents=True, exist_ok=True)
        cand = rd / "candidates.jsonl"
        scores_csv = rd / "dock_scores.csv"
        n_seeds = 0
        if is_round_done(rd):
            recs = gate_and_dedup(read_candidates(cand))
            with open(scores_csv, newline="") as fh:
                scored = {row["smiles"]: float(row["score"]) for row in csv.DictReader(fh)}
        else:
            if arm == "control_b":
                run_pocket_generation(target, cand, ckpt_pocket, n, seed + r)
            else:
                seeds = (select_topk_seeds(all_scores, k) if arm == "treatment"
                         else select_random_seeds(all_scores, k, seed + r))
                n_seeds = len(seeds)
                (rd / "seeds.smi").write_text("\n".join(seeds))
                run_analog_generation(seeds, ckpt_analog, cand)
            recs = gate_and_dedup(read_candidates(cand))
            scored = dock_budget(recs, spec, dock, budget, seed + r)
            with open(scores_csv, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["smiles", "score"])
                for s, v in scored.items():
                    w.writerow([s, v])
        all_scores.update(scored)
        top10 = sorted(scored.values())[:10]
        summary_rows.append({
            "target": target, "arm": arm, "round": r, "n_seeds": n_seeds,
            "n_gated": len(recs), "n_docked": len(scored),
            "best": min(scored.values()) if scored else float("nan"),
            "top10_mean": sum(top10) / len(top10) if top10 else float("nan"),
            "scaffold_div": _scaffold_diversity(list(scored)),
        })
    return select_winners(all_scores, k)


@click.command()
@click.option("--targets", default="data/dock/powered_targets.json")
@click.option("--analog-ckpt", default="data/trained_weights/sf_ed_default.ckpt")
@click.option("--pocket-ckpt", required=True)
@click.option("--arms", default="treatment,control_a,control_b")
@click.option("--rounds", default=2, type=int)
@click.option("--budget", default=60, type=int)
@click.option("--k", default=3, type=int)
@click.option("--n", default=1000, type=int)
@click.option("--final-m", default=10, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--candidates-dir", default="data/dock/candidates_pocket")
@click.option("--out-dir", default="data/dock/sp_f")
@click.option("--limit-targets", default=None, type=int)
@click.option("--work-dir", default="boltz_out/sp_f")
def main(targets, analog_ckpt, pocket_ckpt, arms, rounds, budget, k, n, final_m, seed,
         candidates_dir, out_dir, limit_targets, work_dir):
    import os
    tgts = json.load(open(targets))
    if limit_targets:
        tgts = tgts[:limit_targets]
    arm_list = [a.strip() for a in arms.split(",")]
    rows: list[dict] = []
    for t in tgts:
        tid = t["target_id"]
        spec = prepare_target(t["pdb_id"], f"{work_dir}/holo/{tid}", ligand_resname=t["ligand_resname"])
        r0 = read_candidates_smi(pathlib.Path(candidates_dir) / f"{tid}.txt")
        # dock round 0 ONCE per target (shared baseline); every arm seeds from these scores
        round0_scores = dock_budget([{"smiles": s} for s in r0], spec, dock, budget, seed)
        for arm in arm_list:
            final = run_arm(arm, tid, spec, analog_ckpt, pocket_ckpt, rounds, budget, k, n, seed,
                            out_dir, round0_scores, rows)
            fdir = pathlib.Path(out_dir) / tid / arm
            fdir.mkdir(parents=True, exist_ok=True)
            (fdir / "final_topM.smi").write_text("\n".join(final[:final_m]))
            print(f"  {tid}/{arm}: final top-{final_m} written", flush=True)
    sp = pathlib.Path(out_dir) / "loop_summary.csv"
    os.makedirs(sp.parent, exist_ok=True)
    with open(sp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "arm", "round", "n_seeds", "n_gated",
                                           "n_docked", "best", "top10_mean", "scaffold_div"])
        w.writeheader(); w.writerows(rows)
    print(f"loop_summary.csv written ({len(rows)} rows)", flush=True)


def read_candidates_smi(path) -> list[str]:
    return [ln.strip() for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]


if __name__ == "__main__":
    main()
