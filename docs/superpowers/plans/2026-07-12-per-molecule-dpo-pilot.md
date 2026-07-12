# SP-DPO Per-Molecule Specificity DPO (pilot) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Cheap pilot — does DPO fine-tuning of the SP-C pocket model on per-molecule own-vs-mismatch specificity pairs make its raw samples more target-specific on held-out pockets than the base model? Pre-committed likely-null (Boltz-refuted shape-fit signal; 95% ADMET-undruggable pool).

**Architecture:** (1) generate pocket-conditioned pools **keeping per-molecule route tensors**; (2) dock own+mismatch → per-molecule specificity → winner/loser pairs; (3) DPO-train SP-C (policy vs frozen reference, conditioned `get_log_likelihood` on routes); (4) evaluate held-out (delta vs base + Boltz + ADMET). Spec: `docs/superpowers/specs/2026-07-12-per-molecule-dpo-pilot-design.md`.

## Global Constraints

- **Venvs:** generation + DPO training → `.venv-train` (GPU); docking → `.venv` (smina, `SMINA` env); ADMET → `.venv-admet`; analysis/tests → `.venv`.
- **Route tensors** (`token_types, rxn_indices, reactant_fps, token_padding_mask` + the pocket `code`) are what `get_log_likelihood` consumes — confirmed on `GenerateResult`.
- **DPO:** policy = trainable SP-C copy; reference = frozen SP-C; loss `-logσ(β[(llπ_w−llref_w)−(llπ_l−llref_l)])`, β=0.1, conditioned per pair's pocket.
- **Targets:** family-diverse split (kinase/GPCR/metabolic mix) — ~10 train, ~4 held-out; held-out never in training.
- **Detach** long docking; monitor by coverage/log, NOT `pgrep -f` (self-matches — SP-SC lesson).
- **Pre-committed likely-null**; Boltz spot-check is the honesty gate.
- Commit only task files via explicit `git add`; footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg`.
- SP-C ckpt: `logs/pocket/2607091019-32f2194@powered-specificity/2026_07_09__10_19_15/checkpoints/epoch=1-step=2255.ckpt`.

---

### Task 1: `dpo_loss` + route serialization helpers (`synformer/molopt/dpo.py`)

**Files:** Create `synformer/molopt/dpo.py`, `tests/test_dpo.py`

**Interfaces:**
- `dpo_loss(llpi_w, llpi_l, llref_w, llref_l, beta=0.1) -> torch.Tensor` — standard DPO loss (mean over pairs) `-logσ(β·[(llpi_w−llref_w)−(llpi_l−llref_l)])`.
- `routes_from_result(result) -> list[dict]` — per-molecule route tensors from a `GenerateResult`: for each batch row i, `{token_types, rxn_indices, reactant_fps, token_padding_mask}` (CPU tensors), plus the shared pocket `code`/`code_padding_mask` stored once by the caller. (Pure w.r.t. a GenerateResult-like object.)

- [ ] **Step 1: Failing test**

```python
# tests/test_dpo.py
import torch
from synformer.molopt.dpo import dpo_loss


def test_dpo_loss_lower_when_policy_prefers_winner():
    # reference indifferent; policy raises winner, lowers loser -> loss should drop
    llref_w = torch.tensor([0.0]); llref_l = torch.tensor([0.0])
    good = dpo_loss(torch.tensor([2.0]), torch.tensor([-2.0]), llref_w, llref_l, beta=0.5)
    bad  = dpo_loss(torch.tensor([-2.0]), torch.tensor([2.0]), llref_w, llref_l, beta=0.5)
    assert good.item() < bad.item()


def test_dpo_loss_reference_cancels():
    # equal policy margins but shifted reference -> loss depends on (policy - reference) margin
    l1 = dpo_loss(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0)
    l2 = dpo_loss(torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]), beta=1.0)
    assert abs(l1.item() - l2.item()) < 1e-5   # same (policy-ref) margins -> same loss


def test_dpo_loss_positive_and_finite():
    v = dpo_loss(torch.tensor([0.5, -0.5]), torch.tensor([0.0, 0.0]),
                 torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), beta=0.1)
    assert torch.isfinite(v) and v.item() > 0
```

- [ ] **Step 2: Run → fail** — `.venv/bin/python -m pytest tests/test_dpo.py -q` (ModuleNotFoundError).

- [ ] **Step 3: Implement**

```python
# synformer/molopt/dpo.py
"""DPO for the pocket-conditioned SynFormer: preference loss on generation routes."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def dpo_loss(llpi_w, llpi_l, llref_w, llref_l, beta: float = 0.1):
    """Standard DPO loss (mean over pairs). ll* are per-pair total log-likelihoods."""
    pi_margin = llpi_w - llpi_l
    ref_margin = llref_w - llref_l
    return -F.logsigmoid(beta * (pi_margin - ref_margin)).mean()


def routes_from_result(result) -> list[dict]:
    """Slice per-molecule route tensors from a batched GenerateResult (CPU)."""
    n = result.token_types.size(0)
    out = []
    for i in range(n):
        out.append({
            "token_types": result.token_types[i:i+1].cpu(),
            "rxn_indices": result.rxn_indices[i:i+1].cpu(),
            "reactant_fps": result.reactant_fps[i:i+1].cpu(),
            "token_padding_mask": result.token_padding_mask[i:i+1].cpu(),
        })
    return out
```

- [ ] **Step 4: Run → pass** (3 tests). **Step 5: Commit** (`synformer/molopt/dpo.py`, `tests/test_dpo.py`).

---

### Task 2: Generation-with-routes (`scripts/generate_routes.py`, `.venv-train`)

**Files:** Create `scripts/generate_routes.py`, `tests/test_generate_routes.py`

**Interfaces:** CLI `generate_routes --ckpt --target --pocket-dir --n --out-prefix` → writes `<prefix>.smi` (SMILES, one/line) + `<prefix>.routes.pt` (`torch.save` of `{code, code_padding_mask, mols:[{smiles, token_types, rxn_indices, reactant_fps, token_padding_mask}]}`). Pure helper `dedup_keep_first(records)` TDD'd; the generation uses `sample_pocket` (returns `(info, result)`; keep `result` for `routes_from_result` and `info`'s depth-1 stacks for SMILES) — mirror `generate_enriched`'s `info`-handling.

- [ ] Steps: failing test on `dedup_keep_first` + route-record shape (mock `GenerateResult`); implement (import torch/model in `main`); GPU smoke on the box (generate 8, confirm `.routes.pt` loads and route tensors match `get_log_likelihood`'s expected args); commit.

*(Key correctness: the SMILES↔route correspondence must be exact — the i-th route is the i-th molecule's. Assert alignment in the smoke by reconstructing one molecule from its stack.)*

---

### Task 3: Preference-pair builder (`scripts/dpo_pairs.py`)

**Files:** Create `scripts/dpo_pairs.py`, `tests/test_dpo_pairs.py`

**Interfaces:**
- `per_molecule_specificity(scores_df, target) -> dict[smiles,float]` — from a docked frame (`molecule, pocket, score`), for source `target`: `spec(m) = z_own − mean(z_mismatch)` per molecule (z per pocket-column, nan-aware). Pure, TDD.
- `make_pairs(spec_by_smiles, frac=0.3) -> list[tuple[str,str]]` — winners = top `frac` by specificity, losers = bottom `frac`; pair them (crossed). Pure, TDD.
- Orchestration (ops): for each train target, dock its `generate_routes` SMILES own + K=12 mismatch (reuse `synformer.dock`, `_sample_mismatch`), build `per_molecule_specificity` → `make_pairs` → write `pairs_<target>.json` (winner/loser SMILES).

- [ ] Steps: failing tests (specificity math on a constructed frame; pairing picks correct top/bottom); implement; commit. (Docking is the ops step in Task 4.)

---

### Task 4: `dpo_train.py` + pilot run + held-out eval + results + finish

**Files:** Create `scripts/dpo_train.py`, `tests/test_dpo_train.py` (smoke of one train step on a tiny toy); `docs/SP_DPO_RESULTS.md`

- [ ] **Step 1: `dpo_train.py`** — load SP-C as policy (trainable) + reference (frozen copy); for each pair, load winner/loser route tensors + the target's pocket `code`; compute `llπ`/`llref` via `model.get_log_likelihood(code=code, code_padding_mask=..., token_types=route.token_types, rxn_indices=..., reactant_fps=..., token_padding_mask=...)`; `dpo_loss` → backprop policy only; Adam, small LR (1e-5), few epochs; save DPO'd ckpt. Log train loss + a KL-to-reference proxy (monitor collapse). TDD: a one-step smoke on a 2-pair toy (mock model returning fixed ll) that loss decreases.

- [ ] **Step 2: Pilot data** (ops, box): pick family-diverse split (~10 train, ~4 held-out from the 41; list them). `generate_routes` for the 10 train targets (GPU). Dock pairs (`dpo_pairs` orchestration, ~5.2k docks, detached, monitor by coverage). Build `pairs_<target>.json`.

- [ ] **Step 3: Train** (`.venv-train`, GPU): `dpo_train.py` on the pooled pairs → DPO'd ckpt. Watch loss + KL.

- [ ] **Step 4: Held-out eval:** sample DPO'd model AND base SP-C on the ~4 held-out pockets (`generate_routes`/`sample_pocket`), dock own + mismatch, compute family-stratified own-vs-mismatch delta **DPO vs base** (reuse `powered_analyze._delta_win_from_matrix`); **Boltz spot-check** (`sp_f_boltz`-style) on DPO held-out top-M; **ADMET** (`admet_score`) on DPO vs base samples.

- [ ] **Step 5: `docs/SP_DPO_RESULTS.md`** — held-out delta (DPO vs base, family-stratified) + Boltz corroboration + ADMET change; honest verdict (did weight-updates confer targeting beyond base? does Boltz corroborate? — vs the pre-committed likely-null). **Decision:** if DPO>base and Boltz-corroborated → recommend full ~7-day run; else capstone-null. Update ledger; commit.

- [ ] **Step 6: Finish branch** — superpowers:finishing-a-development-branch (merge to `powered-specificity`, push fork).

---

## Self-review notes
- **Load-bearing new code = `dpo_loss` (T1) + `dpo_train` loop (T4)**; both TDD'd on toy tensors before any GPU run.
- **SMILES↔route alignment (T2)** is the subtle correctness risk — assert it in the smoke.
- Docking (T3/T4 ops) is the ~1-day compute; monitor by coverage, single clean process (SP-SC lesson: don't relaunch overlapping shards).
- Held-out eval reuses `powered_analyze`/`sp_f_boltz`/`admet_score` — no new analysis code.
- Pre-committed likely-null; the pilot is a decision gate for the full run.
