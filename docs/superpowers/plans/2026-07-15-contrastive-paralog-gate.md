# Contrastive Paralog-Discrimination Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use checkbox syntax.

**Goal:** Run the cheap gate — does a short contrastive fine-tune make the pocket model's route-LL *transferably* discriminate paralogs on a HELD-OUT kinase family (CSNK1) it never trained on?

**Architecture:** Reuse `ll_target_specificity.build_batch` (pathway→route tensors) + `build_pocket_feat` (pocket fields) → `get_log_likelihood_shortcut` for route-LL under a pocket. Contrastive margin loss pushes LL(route|measured-binder isoform) above LL(route|measured-non-binder sibling), trained on MAPK/CDK/PRKC families, tested on held-out CSNK1.

**Tech Stack:** torch (`.venv-train`, GPU), KIBA (already fetched: `data/dock/davis/kiba_routed.csv`, `kiba_acc2gene.json`), `filtered_pathways_370000.pth`, `fpindex.pkl`, pocket `.npz` set.

## Global Constraints

- Route-LL for (route, pocket) = `model.get_log_likelihood_shortcut(batch)["total"].sum()` where `batch` = pocket fields (from `build_pocket_feat(pockets[tid], repeat=1, device)`) + route tensors (from a `build_batch`-style featurization of `pathways[canonical_smiles]` using `fpindex._fp`). Pattern verbatim from `scripts/ll_target_specificity.py:37-49`.
- KIBA binder ≥ 12.1, non-binder ≤ 11.3, else drop.
- Train families = MAPK, CDK, PRKC; **held-out family = CSNK1** (no CSNK1 isoform ever in training).
- Family→isoform→UniProt map from `data/dock/davis/kiba_acc2gene.json` (acc→gene) inverted; pocket target_id = `<UniProt>_WT`; pocket features from `load_pockets("data/pockets")`.
- Molecule identity = canonical SMILES (RDKit) for route lookup into `filtered_pathways`.
- GPU runs detached via `nohup … &` inside a background Bash call; NEVER `pkill -f <scriptname>` in a launch command (self-match → SIGTERM 144). Save ckpts via `build_out_checkpoint` (from `scripts/dpo_train.py`).
- Commit only task files (explicit `git add`); footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg`.

---

### Task 1: `contrastive_data.py` — labels, family split, within-family triples

**Files:** Create `scripts/contrastive_data.py`, `tests/test_contrastive_data.py`

**Produces:** `binder_label(kiba, bind=12.1, non=11.3)->str|None`; `make_within_family_triples(rows, gene2fam, train_fams)->list[(smiles, binder_gene, nonbinder_gene, fam)]`; a `main` that writes `data/dock/contrastive/{train_triples.json, heldout_triples.json, gene2tid.json}`.

- [ ] **Step 1: failing tests**
```python
# tests/test_contrastive_data.py
from scripts.contrastive_data import binder_label, make_within_family_triples

def test_binder_label():
    assert binder_label(12.5) == "bind"
    assert binder_label(11.0) == "non"
    assert binder_label(11.7) is None   # ambiguous middle dropped

def test_make_within_family_triples_train_only_and_within_family():
    # drug X: binds MAPK1 (12.5), not MAPK3 (11.0); CSNK1 held out
    rows = [{"smiles": "X", "gene": "MAPK1", "kiba": 12.5},
            {"smiles": "X", "gene": "MAPK3", "kiba": 11.0},
            {"smiles": "X", "gene": "CSNK1D", "kiba": 12.5},
            {"smiles": "X", "gene": "CSNK1E", "kiba": 11.0}]
    gene2fam = {"MAPK1": "MAPK", "MAPK3": "MAPK", "CSNK1D": "CSNK1", "CSNK1E": "CSNK1"}
    tr = make_within_family_triples(rows, gene2fam, train_fams={"MAPK"})
    assert ("X", "MAPK1", "MAPK3", "MAPK") in tr
    assert all(t[3] == "MAPK" for t in tr)          # only train family
    assert not any("CSNK1" in t[1] or "CSNK1" in t[2] for t in tr)  # held-out excluded
```
- [ ] **Step 2: run → fail.**
- [ ] **Step 3: implement** `binder_label` (thresholds) and `make_within_family_triples` (group rows by (smiles, fam); within each, for binder×non-binder gene pairs where fam∈train_fams, emit (smiles, b_gene, nb_gene, fam)); `main` builds `gene2fam` from KIBA family regexes (reuse the patterns from `scratchpad/kiba_paralog.py`) + `gene2tid` from `kiba_acc2gene.json` inverted (gene→`<acc>_WT`), reads `kiba_routed.csv`, filters to rows whose canonical SMILES ∈ `filtered_pathways` AND whose gene has a pocket `.npz`, and writes train (MAPK/CDK/PRKC) + heldout (CSNK1) triples.
- [ ] **Step 4: run → pass. Step 5: run `main` live** (`.venv/bin/python -m scripts.contrastive_data`), confirm nonzero train + heldout(CSNK1) triples. **Step 6: commit.**

---

### Task 2: `contrastive_train.py` — margin loss + short fine-tune

**Files:** Create `scripts/contrastive_train.py`, `tests/test_contrastive_train.py`

**Consumes:** triples (Task 1); `build_out_checkpoint` (`scripts/dpo_train.py`); pocket/route featurization (Global Constraints). **Produces:** `contrastive_loss(ll_bind, ll_nonbind, margin=2.0)->Tensor`; fine-tuned ckpt `data/ckpt/contrastive_pilot.ckpt`.

- [ ] **Step 1: failing test**
```python
# tests/test_contrastive_train.py
import torch
from scripts.contrastive_train import contrastive_loss

def test_contrastive_loss_rewards_binder_over_nonbinder():
    good = contrastive_loss(torch.tensor([3.0]), torch.tensor([0.0]), margin=2.0)  # bind >> non
    bad  = contrastive_loss(torch.tensor([0.0]), torch.tensor([3.0]), margin=2.0)  # inverted
    assert good.item() < bad.item()
    assert torch.isfinite(good) and good.item() >= 0
```
- [ ] **Step 2: run → fail.**
- [ ] **Step 3: implement** `contrastive_loss = softplus(margin - (ll_bind - ll_nonbind)).mean()`; a `route_pocket_ll(model, pathway, pocket, fp, device)` helper (build route tensors à la `ll_target_specificity.build_batch` + `build_pocket_feat(pocket,1,device)` → `get_log_likelihood_shortcut`); `main` loads SP-C (trainable) via `load_model`, loads `pockets`, `pathways`, `fpindex._fp`, iterates train triples computing `ll_bind`/`ll_nonbind` (grad on), `contrastive_loss`, AdamW lr 1e-5, ~1–3 passes; logs mean train margin + a held-out-family margin monitor each N steps; saves `build_out_checkpoint`.
- [ ] **Step 4: run → pass (loss test). Step 5: commit** (run of `main` is the ops step in Task 4).

---

### Task 3: `discrim_eval.py` — held-out-family paralog win-rate

**Files:** Create `scripts/discrim_eval.py`, `tests/test_discrim_eval.py`

**Consumes:** heldout triples (Task 1); `route_pocket_ll` (Task 2); base SP-C + fine-tuned ckpts. **Produces:** `winrate(triple_lls)->float`; clustered bootstrap CI; prints base vs FT held-out CSNK1 win-rate.

- [ ] **Step 1: failing test**
```python
# tests/test_discrim_eval.py
from scripts.discrim_eval import winrate
def test_winrate():
    # (ll_bind, ll_nonbind) per triple: 2 correct, 1 wrong -> 2/3
    assert abs(winrate([(1.0, 0.0), (2.0, 1.0), (0.0, 1.0)]) - 2/3) < 1e-9
```
- [ ] **Step 2: run → fail. Step 3: implement** `winrate` (fraction ll_bind>ll_nonbind); `main` computes, for base and FT models, held-out-CSNK1 triple LLs via `route_pocket_ll`, `winrate` for each, and a **molecule-clustered bootstrap** (resample distinct SMILES) CI for FT and for (FT−base); prints base vs FT vs chance(0.5). **Step 4: run → pass. Step 5: commit.**

---

### Task 4: Run the gate, results, verdict

- [ ] **Step 1:** `.venv/bin/python -m scripts.contrastive_data`; confirm train + CSNK1 heldout triple counts (expect train ≫ heldout ~ handful).
- [ ] **Step 2:** short FT — `nohup .venv-train/bin/python -m scripts.contrastive_train … > logs/contrastive/train.log 2>&1 &` (GPU, detached, watch by coverage). Verify train margin rises + save ckpt.
- [ ] **Step 3:** `.venv-train/bin/python -m scripts.discrim_eval` → base vs FT held-out CSNK1 win-rate + clustered CI.
- [ ] **Step 4:** Write `docs/CONTRASTIVE_DISCRIM_RESULTS.md` — pre-committed verdict: **PASS** (held-out CSNK1 win-rate > chance AND > base, clustered CI excludes 0.5) → recommend full contrastive pretraining; **FAIL** (train rises, held-out at chance = SP-DPO pattern) → stop. Report train-family vs held-out side by side; note small-N (rotate held-out family if borderline).
- [ ] **Step 5:** Advisor-review the verdict; update `FINDINGS.md` + `CAPSTONE.md`; commit; push; update `.superpowers/sdd/progress.md`.

---

## Self-review notes
- **New code = 3 pure functions** (`binder_label`, `make_within_family_triples`, `contrastive_loss`, `winrate`) all TDD'd; route/pocket LL and train loop reuse `ll_target_specificity.build_batch` + `build_pocket_feat` + `get_log_likelihood_shortcut` + `dpo_train.build_out_checkpoint`.
- **Spec coverage:** T1 = §3 data/labels/split; T2 = §4 objective; T3 = §5 transfer gate; T4 = §6 decision + results.
- **Honesty:** held-out CSNK1 is small (7 disc. drugs) — a null is "no signal at this scale"; report train-vs-heldout gap; rotate held-out family for robustness if borderline.
- **Reuse risk:** confirm `get_log_likelihood_shortcut` accepts a pocket batch (it calls `encode`, which the pocket encoder reads `pocket_ca/pocket_restype` from) — verify on first live LL call before the full run.
