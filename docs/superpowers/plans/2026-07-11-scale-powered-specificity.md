# SP-SC Scale Powered Specificity (N=67, sampled mismatch) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompute the own-vs-mismatch normalized-delta specificity readout over N=67 pocket-ready drug-like-holo test targets with sampled mismatch (own + K=12 random pockets), and compare to the N=20 baseline.

**Architecture:** One small code change (`powered_run --mismatch-sample K`) plus ops: select the 67-target set, pocket-generate candidates for the 47 new targets, run sharded sampled-mismatch docking (~36h), and analyze with the unchanged (nan-aware) `powered_analyze`.

**Tech Stack:** Python 3.10, click, pandas, numpy, biotite, smina; GPU for pocket generation. Spec: `docs/superpowers/specs/2026-07-11-scale-powered-specificity-design.md`.

## Global Constraints

- **smina:** `export SMINA="$(pwd)/smina.static"` before docking.
- **Detach** long jobs (`setsid … nohup … </dev/null &`); box has no reaper. Docking is CPU-bound (~4 concurrent, ~240/hr).
- **GPU generation → `.venv-train`; docking/target-selection/analysis/tests → `.venv`.** Pocket candidates only (full embeddings absent).
- **Network (RCSB/target-vetting)** goes through the proxy (`https_proxy=http://proxy.unimaas.nl:3128/`).
- **N=67** = 20 current (reuse `candidates_pocket/`) + 47 new pocket-ready; the 9 pocketless of the 76 are excluded.
- **Sampled mismatch:** dock each source's top-M into its own pocket + K=12 seeded-random mismatch pockets. `powered_analyze` reused UNCHANGED (nan-aware per-column z).
- **SP-C checkpoint:** `logs/pocket/2607091019-32f2194@powered-specificity/2026_07_09__10_19_15/checkpoints/epoch=1-step=2255.ckpt`.
- **Commits:** author `michel.dumontier@gmail.com`; footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session:` line. Commit only task files via explicit `git add <path>` — never `git add -A`.

---

### Task 1: `powered_run --mismatch-sample K`

**Files:**
- Modify: `scripts/powered_run.py` (add helper + option; edit the crystal mismatch loop at lines ~181–187)
- Test: `tests/test_powered_run_sample.py`

**Interfaces:**
- Produces:
  - `_sample_mismatch(tid: str, ok_ids: list[str], k: int, seed: int) -> list[str]` — returns `[tid] + k` distinct pockets sampled uniformly at random (seeded by `(seed, tid)`) from `ok_ids` excluding `tid`; if `k >= len(others)`, returns `[tid] + all others` (degenerates to all-pairs). `tid` always first, never duplicated.
  - New CLI option `--mismatch-sample` (default `None` ⇒ current full-all-pairs behavior unchanged).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_powered_run_sample.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_powered_run_sample.py -q`
Expected: FAIL (`ImportError: cannot import name '_sample_mismatch'`).

- [ ] **Step 3: Write minimal implementation** — add the helper near the top of `scripts/powered_run.py` (after imports):

```python
import random


def _sample_mismatch(tid, ok_ids, k, seed):
    others = [t for t in ok_ids if t != tid]
    if k >= len(others):
        return [tid] + others
    rng = random.Random((seed, tid))
    return [tid] + rng.sample(others, k)
```

Add the CLI option (near the other `@click.option`s):

```python
@click.option("--mismatch-sample", "mismatch_sample", default=None, type=int,
              help="If set, dock each source's top-M into its OWN pocket + this many random mismatch "
                   "pockets (seeded), instead of ALL pockets. Makes the run linear, not quadratic; "
                   "powered_analyze's nan-aware per-column z handles the resulting sparse matrix.")
```

Add `mismatch_sample` to the `def main(...)` signature (with the other params), then replace the crystal mismatch loop (lines ~181–187):

```python
    for t in sources:
        tid = t["target_id"]
        pockets = _sample_mismatch(tid, ok_ids, mismatch_sample, seed) if mismatch_sample else ok_ids
        for pk in pockets:
            spec_pk = holo[pk]
            for smi in top_m_smiles[tid]:
                _dock_into(dock_fn, spec_pk, smi, seed, tid, pk, "candidate", scores, done)
        print(f"  {tid}: crystal mismatch done ({len(pockets)} pockets)", flush=True)
```

(Leave the AF arm untouched — this pass is crystal-only.)

- [ ] **Step 4: Run tests + a focused regression**

Run: `.venv/bin/python -m pytest tests/test_powered_run_sample.py -q` (expect 3 pass)
Run: `.venv/bin/python -m pytest tests/ -q` (full suite stays green — `mismatch_sample=None` default keeps existing behavior)

- [ ] **Step 5: Commit**

```bash
git add scripts/powered_run.py tests/test_powered_run_sample.py
git commit -m "feat(SP-SC): powered_run --mismatch-sample K (own + K seeded-random pockets)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 2: Select the 67-target set

**Files:**
- Create: `data/dock/powered_targets_67.json`

- [ ] **Step 1: Build the pocket-ready candidate accession list** and run `powered_targets` restricted to it, N=67:

```bash
cd ~/pw && export https_proxy=http://proxy.unimaas.nl:3128/ http_proxy=http://proxy.unimaas.nl:3128/
# powered_targets vets drug-like single holo via RCSB; restrict its pool to pocket-ready accs.
.venv/bin/python -c "
import json, pandas as pd, glob, os
pool=set(json.load(open('data/dock/druglike_holo_accs.json')))
test=set(t.split('_')[0] for t in pd.read_csv('data/protein_molecule_pairs/sp2_test.csv').target_id.unique())
pockets=set(os.path.basename(p)[:-4] for p in glob.glob('data/pockets/*.npz'))
ready=sorted(a for a in (pool&test) if f'{a}_WT' in pockets)
json.dump(ready, open('data/dock/druglike_holo_pocketready.json','w'))
print('pocket-ready accs:', len(ready))
"
.venv/bin/python -m scripts.powered_targets --pool data/dock/druglike_holo_pocketready.json \
  --n-target 67 --over-select 76 --out data/dock/powered_targets_67.json
```

- [ ] **Step 2: Verify** the JSON: 67 (or however many vetted; some new PDBs may fail drug-like-single-holo vetting → fewer), each with `target_id`/`pdb_id`/`ligand_resname`, and all 20 current targets present (superset). Record the actual N. If vetting drops many, N<67 — that's fine, log it.

- [ ] **Step 3: Commit** the target JSON.

```bash
git add data/dock/powered_targets_67.json
git commit -m "data(SP-SC): 67-target (pocket-ready) powered specificity set

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

*(Note: `data/` is gitignored → this commit will be a no-op/refused; keep the JSON on the share and record its path + N in the ledger instead.)*

---

### Task 3: Generate pocket candidates for the new targets

- [ ] **Step 1: Identify the new targets** (in `powered_targets_67.json` but lacking `candidates_pocket/<target_id>.txt`).

- [ ] **Step 2: Generate** (detached, GPU 0, `.venv-train`):

```bash
cd ~/pw
CKPT="logs/pocket/2607091019-32f2194@powered-specificity/2026_07_09__10_19_15/checkpoints/epoch=1-step=2255.ckpt"
setsid env CUDA_VISIBLE_DEVICES=0 nohup .venv-train/bin/python -m scripts.dock_prepare generate-pocket \
  --ckpt "$CKPT" --targets data/dock/powered_targets_67.json --pocket-dir data/pockets \
  --candidates-dir data/dock/candidates_pocket --repeat 64 --n-calls 3 --target-min 150 --seed 42 \
  > logs/sc_gen.log 2>&1 </dev/null &
# generate-pocket skips targets that already have candidate files (idempotent) -> only the 47 new run.
```

- [ ] **Step 3: On completion, verify** every target in the JSON has a `candidates_pocket/<target_id>.txt` with ≥~140 unique SMILES; log any that under-generated or lacked a pocket.

---

### Task 4: Sampled-mismatch docking run

- [ ] **Step 1: Launch** (detached, ~36h; sharded for parallelism — smina caps ~4 concurrent, so run ~4 source-shards each docking own+K into its sampled pockets):

```bash
cd ~/pw && export SMINA="$(pwd)/smina.static"
for i in 0 1 2 3; do
  setsid nohup .venv/bin/python -m scripts.powered_run \
    --targets data/dock/powered_targets_67.json \
    --candidates-dir data/dock/candidates_pocket \
    --scores data/dock/dock_scores_scale_shard$i.csv \
    --matrix-out data/dock/matrix_scale_shard$i.json \
    --n-candidates 150 --n-refs 0 --top-m 10 --seed 42 \
    --mismatch-sample 12 --source-shard $i/4 --work-dir boltz_out/sc_$i \
    > logs/sc_dock_$i.log 2>&1 </dev/null &
done
# monitor: grep -c "mismatch done" logs/sc_dock_*.log ; cut -d" " -f1-3 /proc/loadavg
```

- [ ] **Step 2: On completion, merge** the shard score CSVs (dedup by `(molecule, pocket)`) into `data/dock/dock_scores_scale.csv`; build the merged `matrix_scale.json` (union of shard target lists). Verify per-source pocket counts ≈ 13 (own + 12).

*Note:* own-pocket docking for each target is done by whichever shard owns it (`--n-refs 0` skips known/random; each shard's own-pocket phase covers its sources). Confirm every source has its own-pocket (diagonal) score before analysis.

---

### Task 5: Analyze + results + finish

- [ ] **Step 1: Run `powered_analyze`** (unchanged) on the merged scores:

```bash
cd ~/pw && .venv/bin/python -m scripts.powered_analyze \
  --scores data/dock/dock_scores_scale.csv --matrix data/dock/matrix_scale.json \
  --n-candidates 150 --top-m 10 --out data/dock/powered_scale_results.json
```

Confirm it reconstructs top-M per target from own-pocket scores and computes the nan-aware per-column-z delta over the sparse matrix. Sanity: report per-pocket finite-column counts (≈13).

- [ ] **Step 2 (controlled baseline — do NOT skip): recompute the N=20 delta under the SAME sampled
  mismatch.** The published N=20 result used FULL all-pairs; comparing it directly to the N=67 sampled
  result is confounded (different mismatch design). Subsample the existing N=20 all-pairs matrix
  (`data/dock/dock_scores_pocket.csv` / `matrix_targets.json`) to own + K=12 seeded-random mismatch per
  source and recompute its delta with the same `_delta_win_from_matrix`. Report **three** numbers: N=20
  full all-pairs (published), N=20 sampled-mismatch (apples-to-apples), N=67 sampled-mismatch. The honest
  comparison is N=67-sampled vs N=20-sampled; N=20-full is context.

- [ ] **Step 3: Write `docs/POWERED_SCALE_RESULTS.md`** — N (actual, after vetting/generation), the three
  deltas above + win-rate + bootstrap CIs; does the specificity signal hold/tighten at 3× targets *under
  the matched sampled-mismatch design*? Method (sampled mismatch K=12, pocket candidates, crystal, 1 seed);
  caveats (67-not-76, sampled columns ~13, pocket-only, 1 seed, sampled-vs-full confound controlled in
  Step 2, any vetting/gen dropouts). Reproduce commands.

- [ ] **Step 4: Honest verdict** — does the modest own-vs-mismatch specificity survive at N≈67 (with tighter CI), strengthen, or wash out? State with numbers, comparing **N=67-sampled vs N=20-sampled** (the matched baseline from Step 2).

- [ ] **Step 5: Update the SDD ledger; commit** results doc.

- [ ] **Step 6: Finish the branch** — superpowers:finishing-a-development-branch (merge to `powered-specificity`, push to fork; box authoritative, reconcile remote first).

---

## Self-review notes

- **Only code change** is Task 1 (`--mismatch-sample`); `powered_analyze` is reused unchanged (verified nan-aware per-column z handles the sparse matrix).
- **`data/` is gitignored** — target JSON, candidates, scores live on the share; commits are code + docs only. Record data artifact paths + actual N in the ledger.
- **N may end < 67** if new-target PDB vetting or pocket generation drops some — the plan logs and proceeds; the readout reports the actual N.
- **Sharding** gives the 4-wide parallelism smina allows; each shard writes its own scores CSV, merged before analysis (dedup by molecule,pocket — the powered study's established pattern).
- **Advisor/uncontrolled-comparison habit:** at write-up, the N=67 vs N=20 comparison differs in mismatch design (sampled vs full all-pairs) — call that out; if in doubt, also compute the N=20 delta under sampled mismatch (subsample the existing all-pairs matrix) for an apples-to-apples baseline.
