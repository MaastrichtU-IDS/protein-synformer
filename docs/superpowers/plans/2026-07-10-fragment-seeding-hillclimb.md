# SP-F Fragment-Seeding Hill-Climb — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a docking-guided local-search loop that seeds SynFormer's analog sampler on the top-k dockers, iterates (hill-climb), and measures — against two budget-matched controls — whether exploring a good binder's synthesizable neighborhood produces better binders (the lever SP-L's motif-enrichment structurally couldn't be).

**Architecture:** Reuse SP-L's `.venv`-side helpers (`gate_and_dedup`, `dock_budget`, `select_winners`) and its two-venv split. New: restore `featurize_stack` (analog sampler dep), an analog-generation subprocess (`.venv-train`, GPU), and a 3-arm hill-climb orchestrator (`.venv`). Analog generation is seed-molecule-conditioned via `sf_ed_default.ckpt`; the pocket enters only through docking-selection of seeds.

**Tech Stack:** Python 3.10, PyTorch, RDKit, biotite, smina, SynFormer analog sampler, click, pandas, pytest. Spec: `docs/superpowers/specs/2026-07-10-fragment-seeding-hillclimb-design.md`.

## Global Constraints

- **Two venvs:** analog/pocket generation → `.venv-train/bin/python` (GPU); docking + orchestration + gate + tests → `.venv/bin/python`.
- **smina:** `export SMINA="$(pwd)/smina.static"` before docking.
- **GPU:** use `CUDA_VISIBLE_DEVICES=0` (GPU 1 is occupied by another user; GPU 0/2 free). Analog worker: `num_gpus=1`.
- **Base checkpoint:** `data/trained_weights/sf_ed_default.ckpt` (already on box; encoder_type `smiles`; loads with 0 missing/unexpected keys). fpindex/matrix at `data/processed/comp_2048/` (referenced by the ckpt config, already resolve).
- **Detach long jobs:** `setsid … nohup … </dev/null &`.
- **Three arms, budget-matched:** `treatment` (analog on top-k best), `control_a` (analog on k uniform-random docked), `control_b` (fresh SP-C pocket draws via `generate_enriched --weights NONE`). All dock `B`/round. Round 0 shared.
- **Shakedown scope:** targets `O43570_WT`, `P06537_WT`; `R=2` rounds; `B=60` docks/round/arm; `k=3` seeds; `M=10` final; analog `search_width=24 exhaustiveness=64 time_limit=180`.
- **Determinism:** seed RNG from `(base_seed, target, arm, round)`. **nan** docks excluded. **Resumable:** never re-dock a completed round.
- **Commits:** author `michel.dumontier@gmail.com`; footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session:` line. Commit ONLY task files via explicit `git add <path>` — the working tree has pre-existing drift; never `git add -A`.
- **Reuse, don't duplicate:** import `gate_and_dedup`, `dock_budget`, `select_winners`, `read_candidates` from `scripts.optimize_loop`; `passes_gate` from `synformer.molopt.enrich`; `dock`/`prepare_target` from `synformer.dock`.

---

### Task 1: Restore `featurize_stack` in `synformer/data/common.py`

**Files:**
- Modify: `synformer/data/common.py` (un-comment `featurize_stack_actions` + `featurize_stack`; the working tree already holds this edit from the Task-0 spike — verify and commit it)
- Test: `tests/test_featurize_stack.py`

**Interfaces:**
- Produces: `featurize_stack(stack: Stack, end_token: bool, fpindex: FingerprintIndex) -> dict[str, torch.Tensor]` and `featurize_stack_actions(mol_idx_seq, rxn_idx_seq, end_token, fpindex) -> dict` — importable from `synformer.data.common` (the analog sampler's `state_pool.py` imports them).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_featurize_stack.py
import torch
from synformer.data.common import TokenType, featurize_stack_actions


def test_featurize_stack_actions_marks_reaction_and_reactant_tokens():
    # one reaction at step 1, one reactant (building block) at step 2
    class FakeFpindex:
        class _opt: dim = 4
        fp_option = _opt()
        def __getitem__(self, i):  # returns (mol, fp)
            import numpy as np
            return None, np.ones(4, dtype="float32") * i
    feats = featurize_stack_actions(
        mol_idx_seq=[None, 7], rxn_idx_seq=[3, None], end_token=False, fpindex=FakeFpindex()
    )
    assert feats["token_types"][0] == TokenType.START
    assert feats["token_types"][1] == TokenType.REACTION
    assert feats["rxn_indices"][1] == 3
    assert feats["token_types"][2] == TokenType.REACTANT
    assert torch.allclose(feats["reactant_fps"][2], torch.full((4,), 7.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_featurize_stack.py -q`
Expected: FAIL with `ImportError: cannot import name 'featurize_stack_actions'` (functions still commented out if the working-tree edit was reverted) — OR pass if the spike edit is present. If it passes immediately, confirm the edit is the spike's un-comment (diff `synformer/data/common.py`), then continue.

- [ ] **Step 3: Ensure the restore is in place**

In `synformer/data/common.py`, the two functions `featurize_stack_actions` and `featurize_stack` must be un-commented (they sit between the `ProjectionBatch` class and `create_data`, which stays inside a `'''…'''` block). The Task-0 spike already made this edit; verify with:
`.venv/bin/python -c "from synformer.data.common import featurize_stack, featurize_stack_actions; print('ok')"`

- [ ] **Step 4: Run test + a focused regression to verify nothing else broke**

Run: `.venv/bin/python -m pytest tests/test_featurize_stack.py -q` (expect PASS)
Run: `.venv/bin/python -m pytest tests/ -q` (expect the full suite still green — the change only re-adds two functions)

- [ ] **Step 5: Commit**

```bash
git add synformer/data/common.py tests/test_featurize_stack.py
git commit -m "fix(SP-F): restore featurize_stack(_actions) for the analog sampler

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 2: Analog generation script (`scripts/generate_analogs.py`, `.venv-train`)

**Files:**
- Create: `scripts/generate_analogs.py`
- Test: `tests/test_generate_analogs.py`

**Interfaces:**
- Consumes: `synformer.sampler.analog.parallel.run_parallel_sampling_return_smiles`, `synformer.chem.mol.Molecule`.
- Produces: CLI `generate_analogs --seeds <smi-file> --model <ckpt> --out <jsonl> [--search-width 24 --exhaustiveness 64 --time-limit 180 --num-gpus 1]`; writes one JSON per unique analog: `{"smiles": str, "seed": str, "sim": float}`.
  - `read_seeds(path) -> list[str]` — one SMILES per line, blank lines skipped.
  - `df_to_records(df) -> list[dict]` — from the analog DataFrame (columns `smiles`,`target`,`score`) to deduped `{"smiles","seed","sim"}` records (dedup by analog smiles, keep highest sim).

- [ ] **Step 1: Write the failing test** (pure helpers; no GPU)

```python
# tests/test_generate_analogs.py
import pandas as pd
from scripts.generate_analogs import read_seeds, df_to_records


def test_read_seeds_skips_blanks(tmp_path):
    p = tmp_path / "s.smi"; p.write_text("CCO\n\n  \nc1ccccc1\n")
    assert read_seeds(str(p)) == ["CCO", "c1ccccc1"]


def test_df_to_records_dedups_by_analog_keeping_best_sim():
    df = pd.DataFrame({
        "smiles": ["CCO", "CCO", "CCN"],
        "target": ["seedA", "seedA", "seedB"],
        "score":  [0.4, 0.9, 0.5],
    })
    recs = df_to_records(df)
    by = {r["smiles"]: r for r in recs}
    assert set(by) == {"CCO", "CCN"}
    assert by["CCO"]["sim"] == 0.9  # best kept
    assert by["CCN"]["seed"] == "seedB"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generate_analogs.py -q`
Expected: FAIL (`ModuleNotFoundError: scripts.generate_analogs`).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/generate_analogs.py
"""Generate synthesizable analogs of seed molecules with SynFormer's analog sampler.
Runs in .venv-train (GPU). Emits one JSON record per unique analog:
{"smiles": analog, "seed": seed-it-came-from, "sim": similarity-to-seed}."""
from __future__ import annotations

import json
import pathlib

import click


def read_seeds(path: str | pathlib.Path) -> list[str]:
    return [ln.strip() for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]


def df_to_records(df) -> list[dict]:
    best: dict[str, dict] = {}
    for _, row in df.iterrows():
        smi = row["smiles"]
        sim = float(row["score"])
        if smi not in best or sim > best[smi]["sim"]:
            best[smi] = {"smiles": smi, "seed": str(row["target"]), "sim": sim}
    return list(best.values())


@click.command()
@click.option("--seeds", "seeds_path", required=True)
@click.option("--model", "model_path", required=True)
@click.option("--out", required=True)
@click.option("--search-width", type=int, default=24)
@click.option("--exhaustiveness", type=int, default=64)
@click.option("--time-limit", type=int, default=180)
@click.option("--num-gpus", type=int, default=1)
def main(seeds_path, model_path, out, search_width, exhaustiveness, time_limit, num_gpus):
    from synformer.chem.mol import Molecule
    from synformer.sampler.analog.parallel import run_parallel_sampling_return_smiles

    seeds = read_seeds(seeds_path)
    mols = [Molecule(s) for s in seeds]
    df = run_parallel_sampling_return_smiles(
        input=mols, model_path=pathlib.Path(model_path),
        search_width=search_width, exhaustiveness=exhaustiveness,
        num_gpus=num_gpus, num_workers_per_gpu=1, time_limit=time_limit,
    )
    records = df_to_records(df) if df is not None and len(df) else []
    with open(out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} analog records from {len(seeds)} seeds to {out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generate_analogs.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Box GPU smoke** (`.venv-train`) — reproduce the Task-0 spike through the CLI:

```bash
cd ~/pw && printf 'CC(C)Cc1ccc(cc1)C(C)C(=O)O\n' > /tmp/seed.smi
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python -m scripts.generate_analogs \
  --seeds /tmp/seed.smi --model data/trained_weights/sf_ed_default.ckpt \
  --out /tmp/analogs.jsonl --time-limit 120
head -3 /tmp/analogs.jsonl   # expect JSON records with smiles/seed/sim; dozens+ of analogs
```

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_analogs.py tests/test_generate_analogs.py
git commit -m "feat(SP-F): analog generation script (seeds in, deduped analogs out)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 3: Seed-selection helpers (`scripts/fragment_loop.py`)

**Files:**
- Create: `scripts/fragment_loop.py`
- Test: `tests/test_fragment_loop.py`

**Interfaces:**
- Consumes: `scripts.optimize_loop` (`gate_and_dedup`, `dock_budget`, `select_winners`, `read_candidates`), `random`.
- Produces:
  - `select_topk_seeds(scored: dict[str,float], k: int) -> list[str]` — the k strongest binders (thin wrapper over `select_winners`).
  - `select_random_seeds(scored: dict[str,float], k: int, seed: int) -> list[str]` — k molecules sampled uniformly at random (seeded), among those with non-nan scores.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fragment_loop.py
from scripts.fragment_loop import select_topk_seeds, select_random_seeds


def test_select_topk_seeds_takes_strongest():
    assert select_topk_seeds({"A": -7.0, "B": -3.0, "C": -9.0}, 2) == ["C", "A"]


def test_select_random_seeds_is_seeded_and_sized():
    scored = {c: -float(i) for i, c in enumerate("ABCDEFGH")}
    a = select_random_seeds(scored, 3, seed=1)
    b = select_random_seeds(scored, 3, seed=1)
    c = select_random_seeds(scored, 3, seed=2)
    assert len(a) == 3 and set(a) <= set(scored)
    assert a == b            # deterministic for a fixed seed
    assert a != c or True    # different seed *may* differ; not asserted strictly


def test_select_random_seeds_caps_at_pool_size():
    assert len(select_random_seeds({"A": -1.0, "B": -2.0}, 5, seed=1)) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fragment_loop.py -q`
Expected: FAIL (`ModuleNotFoundError: scripts.fragment_loop`).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/fragment_loop.py (helpers in this task; orchestrator in Task 4)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fragment_loop.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/fragment_loop.py tests/test_fragment_loop.py
git commit -m "feat(SP-F): seed-selection helpers (top-k + seeded random), reuse optimize_loop

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 4: 3-arm hill-climb orchestrator (`scripts/fragment_loop.py`)

**Files:**
- Modify: `scripts/fragment_loop.py`
- Test: `tests/test_fragment_loop.py`

**Interfaces:**
- Consumes: Task-3 helpers; `synformer.dock` (`dock`, `prepare_target`); subprocesses to `scripts.generate_analogs` (analog arms) and `scripts.generate_enriched --weights NONE` (control_b); `enrich._scaffold_diversity`-style metric (compute inline).
- Produces:
  - `run_analog_generation(seeds, model, out, python=".venv-train/bin/python")` and `run_pocket_generation(target, out, ckpt, n, seed, python=".venv-train/bin/python")` — subprocess wrappers (`check=True`).
  - `is_round_done(round_dir)`, `round_dir(base, target, arm, r)` (mirror SP-L).
  - `run_arm(arm, target, spec, ckpt_analog, ckpt_pocket, rounds, budget, k, n, seed, out_dir, round0_scores, summary_rows)` — one arm across rounds; seeded from the SHARED `round0_scores` dict (docked once per target in `main`, not re-docked per arm); `treatment`/`control_a` analog-generate from seeds, `control_b` fresh pocket draws; hill-climb re-seed each round from all-docked (`treatment`: top-k, `control_a`: k-random).
  - CLI `fragment_loop --targets <json> --analog-ckpt <p> --pocket-ckpt <p> --arms treatment,control_a,control_b --rounds 2 --budget 60 --k 3 --n 1000 --final-m 10 --seed 42 --candidates-dir data/dock/candidates_pocket --out-dir data/dock/sp_f --limit-targets N`.
  - Emits `data/dock/sp_f/loop_summary.csv` (`target,arm,round,n_seeds,n_gated,n_docked,best,top10_mean,scaffold_div`) + per-arm `final_topM.smi`.

- [ ] **Step 1: Write the failing test** (generation + dock stubbed)

```python
# append to tests/test_fragment_loop.py
import json, pathlib
import scripts.fragment_loop as fl


def test_is_round_done_requires_nonempty_scores(tmp_path):
    d = tmp_path / "r0"; d.mkdir()
    assert fl.is_round_done(d) is False
    (d / "dock_scores.csv").write_text("smiles,score\nA,-7\n")
    assert fl.is_round_done(d) is True


def test_run_arm_control_b_never_seeds_and_treatment_seeds_topk(tmp_path, monkeypatch):
    calls = {"analog_seeds": [], "pocket": 0}
    def fake_analog(seeds, model, out, python=None):
        calls["analog_seeds"].append(list(seeds))
        pathlib.Path(out).write_text("\n".join(
            json.dumps({"smiles": s, "seed": seeds[0], "sim": 0.5}) for s in ["X", "Y", "Z"]))
    def fake_pocket(target, out, ckpt, n, seed, python=None):
        calls["pocket"] += 1
        pathlib.Path(out).write_text("\n".join(
            json.dumps({"smiles": s, "bb": [1], "tpl": [1]}) for s in ["P", "Q", "R"]))
    monkeypatch.setattr(fl, "run_analog_generation", fake_analog)
    monkeypatch.setattr(fl, "run_pocket_generation", fake_pocket)
    monkeypatch.setattr(fl, "passes_gate", lambda s, sa_max=4.0: True)
    monkeypatch.setattr(fl, "dock", lambda spec, smi, seed=0: {"X": -9.0, "Y": -5.0, "Z": -3.0,
                                                               "P": -8.0, "Q": -4.0, "R": -2.0}[smi])
    # shared round-0 scores (docked once in main, passed into every arm)
    round0_scores = {"S0": -6.0, "S1": -7.0, "S2": -8.5}
    fl.run_arm(arm="control_b", target="T", spec=None, ckpt_analog="a", ckpt_pocket="p",
               rounds=1, budget=3, k=2, n=3, seed=1, out_dir=tmp_path,
               round0_scores=round0_scores, summary_rows=[])
    assert calls["pocket"] == 1 and calls["analog_seeds"] == []      # control_b never analogs
    calls["pocket"] = 0
    fl.run_arm(arm="treatment", target="T", spec=None, ckpt_analog="a", ckpt_pocket="p",
               rounds=1, budget=3, k=2, n=3, seed=1, out_dir=tmp_path,
               round0_scores=round0_scores, summary_rows=[])
    # treatment seeds on the top-2 shared round-0 dockers: S2(-8.5), S1(-7.0)
    assert calls["analog_seeds"] and calls["analog_seeds"][0] == ["S2", "S1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fragment_loop.py -q`
Expected: FAIL (`AttributeError: module 'scripts.fragment_loop' has no attribute 'is_round_done'`).

- [ ] **Step 3: Write minimal implementation** — append to `scripts/fragment_loop.py`:

```python
import csv
import json
import pathlib
import subprocess

import click
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from synformer.dock.dock import dock
from synformer.dock.receptor import prepare_target
from synformer.molopt.enrich import passes_gate


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
        rd = round_dir(out_dir, target, arm, r); rd.mkdir(parents=True, exist_ok=True)
        cand = rd / "candidates.jsonl"; scores_csv = rd / "dock_scores.csv"
        n_seeds = 0
        if is_round_done(rd):
            recs = gate_and_dedup(read_candidates(cand))
            import pandas as pd
            scored = dict(zip(pd.read_csv(scores_csv).smiles, pd.read_csv(scores_csv).score))
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
            (fdir / "final_topM.smi").write_text("\n".join(final[:final_m]))
            print(f"  {tid}/{arm}: final top-{final_m} written", flush=True)
    sp = pathlib.Path(out_dir) / "loop_summary.csv"; os.makedirs(sp.parent, exist_ok=True)
    with open(sp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "arm", "round", "n_seeds", "n_gated",
                                           "n_docked", "best", "top10_mean", "scaffold_div"])
        w.writeheader(); w.writerows(rows)
    print(f"loop_summary.csv written ({len(rows)} rows)", flush=True)


def read_candidates_smi(path) -> list[str]:
    return [ln.strip() for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fragment_loop.py -q`
Expected: PASS (5 tests). Then full suite: `.venv/bin/python -m pytest tests/ -q` (stays green).

- [ ] **Step 5: Box dry-run** — 1 target, tiny budget, all 3 arms, confirm resumability:

```bash
cd ~/pw && export SMINA="$(pwd)/smina.static"
POCKET="logs/pocket/2607091019-32f2194@powered-specificity/2026_07_09__10_19_15/checkpoints/epoch=1-step=2255.ckpt"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m scripts.fragment_loop --pocket-ckpt "$POCKET" \
  --limit-targets 1 --rounds 2 --budget 4 --k 2 --n 12 --final-m 3 --out-dir /tmp/sp_f_dry
# re-run: rounds should idempotent-skip (no re-dock).
```

- [ ] **Step 6: Commit**

```bash
git add scripts/fragment_loop.py tests/test_fragment_loop.py
git commit -m "feat(SP-F): 3-arm hill-climb orchestrator (analog seeding + controls, resumable)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 5: Shakedown run + 3-arm readout

**Files:**
- Create: `scripts/sp_f_analyze.py` (per-arm/round curve + pairwise arm comparison from `loop_summary.csv`)
- Test: `tests/test_sp_f_analyze.py`

- [ ] **Step 1: Launch the shakedown** (detached, ~3 h; GPU 0):

```bash
cd ~/pw && export SMINA="$(pwd)/smina.static"
POCKET="logs/pocket/2607091019-32f2194@powered-specificity/2026_07_09__10_19_15/checkpoints/epoch=1-step=2255.ckpt"
setsid env CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -m scripts.fragment_loop \
  --pocket-ckpt "$POCKET" --targets data/dock/powered_targets.json --limit-targets 2 \
  --arms treatment,control_a,control_b --rounds 2 --budget 60 --k 3 --n 1000 --final-m 10 \
  --out-dir data/dock/sp_f > logs/sp_f_shakedown.log 2>&1 </dev/null &
# monitor: grep -E "final top|written" logs/sp_f_shakedown.log
```

- [ ] **Step 2: Write `sp_f_analyze.py`** — TDD a `compare_arms(loop_summary_df) -> dict` that returns, per target, each arm's final-round `top10_mean` and `best`, and the pairwise deltas (treatment−control_a, control_a−control_b, treatment−control_b). Unit-test on a hand-built DataFrame. **Also assert n_docked parity across arms per round** (the SP-L M2 lesson) and flag if it breaks.

- [ ] **Step 3: On completion, run the analysis** and record the three-way decomposition + the best-binder curve. If treatment separates from controls, note magnitude; if not, that's the result.

- [ ] **Step 4: Commit** analysis code + a small JSON summary.

---

### Task 6: Boltz-2 validation (treatment final top-M)

- [ ] **Step 1:** Only if the shakedown shows a treatment win worth corroborating. Prepare Boltz inputs for the treatment arm's `final_topM.smi` per target (reuse `boltz_matrix_prepare` conventions).
- [ ] **Step 2:** Run `.venv-boltz/bin/python -m scripts.boltz_matrix … --accelerator gpu --no-kernels` (detached). `scripts/boltz_analyze.py` now imports on Py-3.10 (fixed).
- [ ] **Step 3:** Record whether co-folding corroborates the docking win (method-dependent verdict, as SP-L/BOLTZ_VALIDATION_RESULTS.md). If the shakedown is a null, note Boltz was not needed.

---

### Task 7: Results doc + ledger

- [ ] **Step 1:** Write `docs/SP_F_RESULTS.md` — question, method (analog hill-climb, 3 budget-matched arms), the round curve, the three-way decomposition (treatment vs control_a vs control_b), diversity trajectory, Boltz verdict if run, caveats (N=2 shakedown; docking proxy; analog-sampler similarity objective ≠ docking), reproduce commands.
- [ ] **Step 2:** State the honest verdict: does docking-guided local search improve binders over the matched controls, or not (and does it change specificity)?
- [ ] **Step 3:** Update `.superpowers/sdd/progress.md` with an SP-F block.
- [ ] **Step 4: Commit** doc + ledger.

---

## Self-review notes

- **Reuse:** Tasks 3–4 import SP-L's `gate_and_dedup`/`dock_budget`/`select_winners`/`read_candidates` — no duplication of the docking/gate logic.
- **Budget parity (SP-L M2):** Task 5 explicitly asserts `n_docked` parity across arms.
- **control_b uses the pocket model** (fresh draws); treatment/control_a use the base analog model — this asymmetry is intrinsic and is exactly what the three-way decomposition separates.
- **Sync-back:** box is authoritative (per user); commit on a `sp-f-*` branch and merge to `powered-specificity` at finish, then push to origin fork (reconcile remote first, as in SP-L).
