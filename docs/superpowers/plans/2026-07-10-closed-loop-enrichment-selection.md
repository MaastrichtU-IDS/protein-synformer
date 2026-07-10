# SP-L Enrichment-Selection Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a closed loop that selects better from the frozen SP-C pocket model via round-over-round building-block / reaction-template enrichment (no weight updates), and measure whether it confers target-specificity against a docking-budget-matched uniform control.

**Architecture:** Pure enrichment math + a model reweight hook (`enrich_weights` passthrough, default `None` ⇒ unchanged) live in `synformer`. A generation script (`.venv-train`, GPU) emits pocket-conditioned candidates with their building-block/template indices. An orchestrator (`.venv`, CPU) runs each target × arm × round: generate → drug-like/SA gate → dock a budget with smina → select winners → compute next-round weights. Specificity is read out with the existing `powered_run`/`powered_analyze` all-pairs harness; final top-M is validated out-of-loop with Boltz-2.

**Tech Stack:** Python 3, PyTorch, RDKit (+ contrib `sascorer`), biotite, smina (`smina.static`), click, pandas, numpy, pytest. Spec: `docs/superpowers/specs/2026-07-10-closed-loop-enrichment-selection-design.md`.

## Global Constraints

- **Two venvs, do not mix:** GPU generation → `.venv-train/bin/python`; docking + orchestration + enrichment + tests → `.venv/bin/python`. `.venv-train` has no biotite/smina; `.venv` is torch-CPU.
- **smina path:** always `export SMINA="$(pwd)/smina.static"` before any docking, else docks silently return `nan`.
- **Detach long jobs:** `setsid … nohup … </dev/null &` (box has no reaper). Log to a file.
- **Enrichment never updates model weights.** `code` (pocket conditioning) is untouched.
- **`enrich_weights=None` must reproduce baseline sampling exactly** (regression guard for all existing callers of `generate_without_stack`).
- **Docking is the only expensive step** (~240 docks/hr, 4-wide). nan docks are excluded from winners and enrichment stats, never propagated.
- **Determinism:** per-round RNG seeded from `(base_seed, target, arm, round)`; ETKDG conformer seed stays fixed in `synformer.dock.dock`.
- **Pilot scope:** 5 targets `O43570_WT, P06537_WT, P10721_WT, P02753_WT, P0C559_WT`; `R=3` rounds; `B=150` docks/round/arm; `N≈1000` pool/round; winners `k=30`; final `M=10`; arms `{enrich, uniform}`.
- **Commits:** author `michel.dumontier@gmail.com`; footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session:` line. Commit only files you created/modified for the task (the box working tree has pre-existing rsync drift — never `git add -A`).
- **SP-C checkpoint:** `logs/pocket/2607091019-32f2194@powered-specificity/2026_07_09__10_19_15/checkpoints/epoch=1-step=2255.ckpt`.

---

### Task 1: Enrichment weight math (`enrich.py`)

**Files:**
- Create: `synformer/molopt/enrich.py`
- Test: `tests/test_enrich.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces:
  - `@dataclass EnrichWeights: bb: dict[int, float]; tpl: dict[int, float]` (sparse; missing index ⇒ weight 1.0).
  - `molecule_index_sets(mol_idx_seq: list[int | None], rxn_idx_seq: list[int | None]) -> tuple[frozenset[int], frozenset[int]]` — drops `None` and negative sentinels.
  - `compute_enrichment_weights(winners: list[tuple[frozenset[int], frozenset[int]]], pool: list[tuple[frozenset[int], frozenset[int]]], w_max: float = 5.0, eps: float = 1e-3) -> EnrichWeights`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enrich.py
from synformer.molopt.enrich import (
    EnrichWeights, molecule_index_sets, compute_enrichment_weights,
)


def test_molecule_index_sets_drops_none_and_sentinel():
    bb, tpl = molecule_index_sets([5, None, 7, -1], [None, 2, None, 3])
    assert bb == frozenset({5, 7})
    assert tpl == frozenset({2, 3})


def test_enrichment_weight_is_presence_fraction_ratio_clipped():
    # BB 1 appears in 2/2 winners but 1/4 pool -> ratio (1.0)/(0.25)=4.0
    # BB 9 appears in 0 winners -> absent from weights (no promotion)
    winners = [(frozenset({1}), frozenset()), (frozenset({1}), frozenset())]
    pool = [
        (frozenset({1}), frozenset()), (frozenset({1}), frozenset()),
        (frozenset({9}), frozenset()), (frozenset({9}), frozenset()),
    ]
    w = compute_enrichment_weights(winners, pool, w_max=5.0, eps=1e-3)
    assert abs(w.bb[1] - 4.0) < 1e-2
    assert 9 not in w.bb


def test_enrichment_weight_clipped_to_wmax_and_floored_at_one():
    # BB 1: 1.0 winners / tiny pool -> huge ratio -> clipped to w_max
    winners = [(frozenset({1}), frozenset())]
    pool = [(frozenset({2}), frozenset())]  # BB 1 absent from pool
    w = compute_enrichment_weights(winners, pool, w_max=5.0, eps=1e-3)
    assert w.bb[1] == 5.0  # clipped
    # a BB that is rarer in winners than pool is floored at 1.0 (enrichment only promotes)


def test_empty_winners_gives_uniform_weights():
    w = compute_enrichment_weights([], [(frozenset({1}), frozenset())])
    assert w.bb == {} and w.tpl == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_enrich.py -q`
Expected: FAIL with `ModuleNotFoundError: synformer.molopt.enrich`.

- [ ] **Step 3: Write minimal implementation**

```python
# synformer/molopt/enrich.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_enrich.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add synformer/molopt/enrich.py tests/test_enrich.py
git commit -m "feat(SP-L): enrichment weight math (presence-fraction ratio, clip, promote-only)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 2: Reweight log-bias functions (`enrich.py`)

**Files:**
- Modify: `synformer/molopt/enrich.py`
- Test: `tests/test_enrich.py`

**Interfaces:**
- Consumes: `EnrichWeights` (Task 1).
- Produces:
  - `reaction_log_bias(n_templates: int, weights: EnrichWeights | None) -> np.ndarray` — shape `(n_templates,)`, `log(w_tpl[i])` for enriched templates else `0.0`.
  - `reactant_log_bias(retrieved_indices: np.ndarray, weights: EnrichWeights | None) -> np.ndarray` — same shape as `retrieved_indices` (`(bsz, n_retrieved)`), `log(w_bb[idx])` per retrieved BB else `0.0`.
  - Convention: callers add these biases to the **post-temperature** logits (`fp_scores / T_reactant + bias`, `reaction_logits / T_reaction + bias`) so a weight `w` multiplies the sampling probability by `w`, temperature-independent. `weights=None` ⇒ all-zero bias ⇒ exact baseline.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_enrich.py
import numpy as np
from synformer.molopt.enrich import reaction_log_bias, reactant_log_bias


def test_reaction_log_bias_none_is_zero():
    b = reaction_log_bias(5, None)
    assert b.shape == (5,) and np.allclose(b, 0.0)


def test_reaction_log_bias_sets_log_weight_for_enriched_templates():
    w = EnrichWeights(bb={}, tpl={2: np.e})  # log(e)=1
    b = reaction_log_bias(4, w)
    assert np.allclose(b, [0.0, 0.0, 1.0, 0.0])


def test_reactant_log_bias_present_index_gets_weight_absent_is_zero():
    # retrieved BBs: rows are batch, cols are the top-k retrieved for the step
    idx = np.array([[10, 11], [12, 10]])
    w = EnrichWeights(bb={10: np.e}, tpl={})  # only BB 10 up-weighted
    b = reactant_log_bias(idx, w)
    assert np.allclose(b, [[1.0, 0.0], [0.0, 1.0]])


def test_reactant_log_bias_absent_bb_has_no_effect():
    # BB 99 is up-weighted but never retrieved -> bias stays all-zero
    idx = np.array([[10, 11]])
    w = EnrichWeights(bb={99: 5.0}, tpl={})
    assert np.allclose(reactant_log_bias(idx, w), 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_enrich.py -q`
Expected: FAIL with `ImportError: cannot import name 'reaction_log_bias'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to synformer/molopt/enrich.py
import numpy as np


def reaction_log_bias(n_templates: int, weights: "EnrichWeights | None") -> np.ndarray:
    bias = np.zeros(n_templates, dtype=np.float32)
    if weights is None or not weights.tpl:
        return bias
    for i, w in weights.tpl.items():
        if 0 <= i < n_templates:
            bias[i] = np.log(w)
    return bias


def reactant_log_bias(retrieved_indices: np.ndarray, weights: "EnrichWeights | None") -> np.ndarray:
    bias = np.zeros(retrieved_indices.shape, dtype=np.float32)
    if weights is None or not weights.bb:
        return bias
    # vectorised lookup: map each retrieved index to log(w) or 0
    flat = retrieved_indices.reshape(-1)
    out = np.zeros(flat.shape, dtype=np.float32)
    for j, idx in enumerate(flat):
        w = weights.bb.get(int(idx))
        if w is not None:
            out[j] = np.log(w)
    return out.reshape(retrieved_indices.shape)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_enrich.py -q`
Expected: PASS (8 tests total).

- [ ] **Step 5: Commit**

```bash
git add synformer/molopt/enrich.py tests/test_enrich.py
git commit -m "feat(SP-L): reweight log-bias (retrieved-BB selection + global template), None=baseline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 3: SMILES drug-like + SA gate (`enrich.py`)

**Files:**
- Modify: `synformer/molopt/enrich.py`
- Test: `tests/test_enrich.py`

**Interfaces:**
- Consumes: RDKit, contrib `sascorer`, `scripts.dock_prepare.MIN_HEAVY_ATOMS`.
- Produces:
  - `ALLOWED_ELEMENTS: set[str]` = `{"C","N","O","S","P","F","Cl","Br","I","H"}`.
  - `sa_score(smiles: str) -> float` — RDKit synthetic-accessibility (1 easy … 10 hard); `float('inf')` if unparseable.
  - `passes_gate(smiles: str, sa_max: float = 4.0) -> bool` — RDKit-valid AND heavy atoms ≥ `MIN_HEAVY_ATOMS` AND only allowed elements AND `sa_score ≤ sa_max`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_enrich.py
from synformer.molopt.enrich import passes_gate, sa_score


def test_gate_rejects_invalid_smiles():
    assert passes_gate("not_a_smiles") is False


def test_gate_rejects_too_small():
    assert passes_gate("CCO") is False  # 3 heavy atoms < MIN_HEAVY_ATOMS


def test_gate_rejects_disallowed_element():
    # a boron-containing molecule large enough otherwise
    assert passes_gate("B1OC2=CC=CC=C2O1" * 1) is False


def test_gate_accepts_drug_like():
    # ibuprofen: 15 heavy atoms, CHO only, low SA
    assert passes_gate("CC(C)Cc1ccc(cc1)C(C)C(=O)O") is True


def test_sa_score_finite_for_valid():
    assert sa_score("CC(C)Cc1ccc(cc1)C(C)C(=O)O") < 4.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_enrich.py -q`
Expected: FAIL with `ImportError: cannot import name 'passes_gate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to synformer/molopt/enrich.py
import os
import sys

from rdkit import Chem
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

from scripts.dock_prepare import MIN_HEAVY_ATOMS  # noqa: E402

ALLOWED_ELEMENTS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"}


def sa_score(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("inf")
    return float(sascorer.calculateScore(mol))


def passes_gate(smiles: str, sa_max: float = 4.0) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if mol.GetNumHeavyAtoms() < MIN_HEAVY_ATOMS:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ELEMENTS:
            return False
    return sa_score(smiles) <= sa_max
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_enrich.py -q`
Expected: PASS (13 tests total).

- [ ] **Step 5: Commit**

```bash
git add synformer/molopt/enrich.py tests/test_enrich.py
git commit -m "feat(SP-L): SMILES drug-like + SA gate (reuse MIN_HEAVY_ATOMS, rdkit sascorer)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 4: Wire `enrich_weights` through the model + sampling

**Files:**
- Modify: `synformer/models/synformer.py:320-419` (`generate_without_stack`)
- Modify: `scripts/sample_helpers.py` (`sample_pocket`)
- Test: `tests/test_enrich_hook.py`

**Interfaces:**
- Consumes: `reaction_log_bias`, `reactant_log_bias` (Task 2), `EnrichWeights`.
- Produces:
  - `generate_without_stack(..., enrich_weights: EnrichWeights | None = None)` — applies log-bias before the reaction and reactant multinomials.
  - `sample_pocket(..., enrich_weights=None)` — forwards to `generate_without_stack`.

- [ ] **Step 1: Write the failing test** (pure-logic test of the exact bias arithmetic used in the loop — no model needed)

```python
# tests/test_enrich_hook.py
"""The sampling arithmetic the model performs, replicated on tensors, to prove
the enrichment bias reproduces baseline when None and multiplies probability by w otherwise."""
import numpy as np
import torch
import torch.nn.functional as F

from synformer.molopt.enrich import EnrichWeights, reaction_log_bias, reactant_log_bias


def _reaction_probs(logits, T, weights):
    bias = torch.from_numpy(reaction_log_bias(logits.shape[-1], weights)).to(logits)
    return F.softmax(logits / T + bias, dim=-1)


def test_reaction_bias_none_matches_baseline():
    logits = torch.tensor([[1.0, 2.0, 0.5, -1.0]])
    base = F.softmax(logits / 1.0, dim=-1)
    got = _reaction_probs(logits, 1.0, None)
    assert torch.allclose(base, got)


def test_reaction_weight_multiplies_probability():
    logits = torch.zeros(1, 3)  # uniform -> each 1/3
    w = EnrichWeights(bb={}, tpl={0: 2.0})
    probs = _reaction_probs(logits, 1.0, w)
    # unnormalised weights: [2,1,1] -> normalise
    assert torch.allclose(probs, torch.tensor([[0.5, 0.25, 0.25]]), atol=1e-6)


def test_reactant_bias_shapes_and_absent_noop():
    idx = np.array([[3, 4, 5]])
    assert np.allclose(reactant_log_bias(idx, EnrichWeights(bb={99: 5.0}, tpl={})), 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_enrich_hook.py -q`
Expected: PASS for the pure-math asserts (they import only Task-2 functions). If any fail, fix Task 2. This test also documents the arithmetic the model must implement in Step 3.

- [ ] **Step 3: Modify `generate_without_stack`** — add the parameter and apply the bias. Change the signature (line 320) to include `enrich_weights: "EnrichWeights | None" = None`, add the import at top of file (`from synformer.molopt.enrich import reaction_log_bias, reactant_log_bias`), and replace the reaction/reactant sampling blocks (lines 366-381):

```python
            # Reaction (with optional enrichment log-bias, post-temperature)
            rxn_logits = pred.reaction_logits / temperature_reaction
            rxn_bias = torch.from_numpy(
                reaction_log_bias(rxn_logits.shape[-1], enrich_weights)
            ).to(rxn_logits)
            rxn_idx_next = torch.multinomial(
                torch.nn.functional.softmax(rxn_logits + rxn_bias, dim=-1),
                num_samples=1,
            )[..., 0]
            rxn_indices = torch.cat([rxn_indices, rxn_idx_next[..., None]], dim=-1)
            for b, idx in enumerate(rxn_idx_next):
                reactions[b].append(rxn_matrix.reactions[int(idx.item())])

            # Reactant (building block); enrichment biases SELECTION among retrieved BBs
            fp_scores = (
                torch.from_numpy(1.0 / (pred.retrieved_reactants.distance + 1e-4)).to(reactant_fps).reshape(bsz, -1)
            )
            react_logits = fp_scores / temperature_reactant
            react_bias = torch.from_numpy(
                reactant_log_bias(pred.retrieved_reactants.indices.reshape(bsz, -1), enrich_weights)
            ).to(react_logits)
            fp_idx_next = torch.multinomial(
                torch.nn.functional.softmax(react_logits + react_bias, dim=-1),
                num_samples=1,
            )[..., 0]
```

(Leave lines 383-406 — the gather of `fp_next`/`pfp_next`/`ridx_next`/`reactant_next` by `fp_idx_next` — unchanged.)

- [ ] **Step 4: Modify `sample_pocket`** in `scripts/sample_helpers.py` — add `enrich_weights=None` to its signature and pass it into the `model.generate_without_stack(feat, …, enrich_weights=enrich_weights)` call.

- [ ] **Step 5: Guarded integration smoke** — verify the real SP-C model still samples with `None` and that a strong BB weight shifts output. Mark it slow/skippable (GPU + ckpt).

```python
# append to tests/test_enrich_hook.py
import os, pathlib, pytest

CKPT = pathlib.Path("logs/pocket/2607091019-32f2194@powered-specificity/"
                    "2026_07_09__10_19_15/checkpoints/epoch=1-step=2255.ckpt")

@pytest.mark.skipif(not CKPT.exists(), reason="SP-C ckpt not present (run on the box)")
def test_generate_none_is_baseline_smoke():
    # Run in .venv-train on the box:
    #   .venv-train/bin/python -m pytest tests/test_enrich_hook.py::test_generate_none_is_baseline_smoke -q
    import torch
    from scripts.sample_helpers import load_model
    from scripts.dock_prepare import _load_test_targets_with_embeddings  # noqa: F401
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fpindex, rxn = load_model(str(CKPT), None, device)
    assert model is not None  # deeper assertion added during execution once feat plumbing confirmed
```

- [ ] **Step 6: Run tests**

Run (`.venv`): `.venv/bin/python -m pytest tests/test_enrich_hook.py -q` (smoke skips without ckpt/GPU)
Run full regression (`.venv`): `.venv/bin/python -m pytest tests/ -q`
Expected: pure-math tests PASS; existing suite still green (the `enrich_weights=None` default guarantees unchanged behaviour).

- [ ] **Step 7: On the box, run the GPU smoke** (`.venv-train`) and confirm generation still works; deepen the smoke assertion to sample a small batch with `enrich_weights=None` and assert ≥1 valid SMILES.

- [ ] **Step 8: Commit**

```bash
git add synformer/models/synformer.py scripts/sample_helpers.py tests/test_enrich_hook.py
git commit -m "feat(SP-L): enrich_weights hook in generate_without_stack + sample_pocket (None=baseline)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 5: Enriched generation script (`generate_enriched.py`, `.venv-train`)

**Files:**
- Create: `scripts/generate_enriched.py`
- Test: `tests/test_generate_enriched.py`

**Interfaces:**
- Consumes: `sample_helpers.load_model`, `sample_helpers.sample_pocket`, `synformer.data.pocket_io.load_pockets`, `EnrichWeights`, `molecule_index_sets`.
- Produces: CLI
  `generate_enriched --ckpt <p> --target <id> --pocket-dir data/pockets --weights <weights.json|NONE> --n <int> --repeat 64 --seed <int> --out <candidates.jsonl>`
  writing one JSON object per line: `{"smiles": str, "bb": [int...], "tpl": [int...]}` (deduped by canonical SMILES).
  - `load_weights(path: str | None) -> EnrichWeights | None` — reads `{"bb": {"idx": w}, "tpl": {...}}`; `None`/`"NONE"` ⇒ `None`.
  - `stacks_to_records(smiles_list, stacks) -> list[dict]` — pairs each valid stack's canonical SMILES with `molecule_index_sets(stack.get_mol_idx_seq(), stack.get_rxn_idx_seq())`.

- [ ] **Step 1: Write the failing test** (unit — no GPU; tests weights IO + record extraction with a fake stack)

```python
# tests/test_generate_enriched.py
import json
from scripts.generate_enriched import load_weights, stacks_to_records


class FakeStack:
    def __init__(self, smi, mol_idx, rxn_idx):
        self._smi, self._mol_idx, self._rxn_idx = smi, mol_idx, rxn_idx
    def get_one_top(self):
        class M: smiles = self._smi
        return M()
    def get_mol_idx_seq(self): return self._mol_idx
    def get_rxn_idx_seq(self): return self._rxn_idx
    def get_stack_depth(self): return 1


def test_load_weights_none(tmp_path):
    assert load_weights(None) is None
    assert load_weights("NONE") is None


def test_load_weights_parses_int_keys(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"bb": {"5": 2.0}, "tpl": {"3": 1.5}}))
    w = load_weights(str(p))
    assert w.bb == {5: 2.0} and w.tpl == {3: 1.5}


def test_stacks_to_records_extracts_indices():
    stacks = [FakeStack("CCO", [10, None, 11], [None, 2, None])]
    recs = stacks_to_records(stacks)
    assert recs[0]["smiles"] == "CCO"
    assert sorted(recs[0]["bb"]) == [10, 11]
    assert recs[0]["tpl"] == [2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generate_enriched.py -q`
Expected: FAIL with `ModuleNotFoundError: scripts.generate_enriched`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/generate_enriched.py
"""Generate pocket-conditioned candidates with optional enrichment weights.
Runs in .venv-train (GPU). Emits one JSON record per unique valid molecule:
{"smiles", "bb": [building-block fpindex ids], "tpl": [reaction-template ids]}."""
from __future__ import annotations

import json
import pathlib

import click

from synformer.molopt.enrich import EnrichWeights, molecule_index_sets


def load_weights(path: str | None) -> EnrichWeights | None:
    if path is None or path == "NONE":
        return None
    d = json.loads(pathlib.Path(path).read_text())
    return EnrichWeights(
        bb={int(k): float(v) for k, v in d.get("bb", {}).items()},
        tpl={int(k): float(v) for k, v in d.get("tpl", {}).items()},
    )


def stacks_to_records(stacks) -> list[dict]:
    out, seen = [], set()
    for st in stacks:
        if st.get_stack_depth() != 1:
            continue
        smi = st.get_one_top().smiles
        if not smi or smi in seen:
            continue
        seen.add(smi)
        bb, tpl = molecule_index_sets(st.get_mol_idx_seq(), st.get_rxn_idx_seq())
        out.append({"smiles": smi, "bb": sorted(bb), "tpl": sorted(tpl)})
    return out


@click.command()
@click.option("--ckpt", required=True)
@click.option("--target", required=True)
@click.option("--pocket-dir", default="data/pockets")
@click.option("--weights", default="NONE")
@click.option("--n", type=int, default=1000)
@click.option("--repeat", type=int, default=64)
@click.option("--seed", type=int, default=42)
@click.option("--out", required=True)
def main(ckpt, target, pocket_dir, weights, n, repeat, seed, out):
    import torch
    from scripts.sample_helpers import load_model, sample_pocket
    from synformer.data.pocket_io import load_pockets

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fpindex, rxn_matrix = load_model(ckpt, None, device)
    pockets = load_pockets(pocket_dir)
    ew = load_weights(weights)

    records, seen = [], set()
    calls = 0
    while len(records) < n and calls < max(4, (n // repeat) * 3):
        calls += 1
        _info, stacks = sample_pocket(
            target, model, fpindex, rxn_matrix, pockets, device,
            repeat=repeat, enrich_weights=ew,
        )
        for r in stacks_to_records(stacks):
            if r["smiles"] not in seen:
                seen.add(r["smiles"])
                records.append(r)
    with open(out, "w") as fh:
        for r in records[:n]:
            fh.write(json.dumps(r) + "\n")
    print(f"{target}: wrote {min(len(records), n)} records to {out} ({calls} calls)", flush=True)


if __name__ == "__main__":
    main()
```

*Note during execution:* confirm `sample_pocket` returns `(info, stacks)` where `stacks` is the built stack list; if it returns something else, adapt `stacks_to_records`'s input at the call site (the extraction function itself stays pure/tested).

- [ ] **Step 4: Run unit test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generate_enriched.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Box GPU smoke** (`.venv-train`) — one small real generation:

```bash
cd ~/pw && CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python -m scripts.generate_enriched \
  --ckpt "logs/pocket/2607091019-32f2194@powered-specificity/2026_07_09__10_19_15/checkpoints/epoch=1-step=2255.ckpt" \
  --target O43570_WT --n 20 --repeat 32 --seed 42 --out /tmp/ge_smoke.jsonl
head -1 /tmp/ge_smoke.jsonl   # expect a JSON record with smiles/bb/tpl
```

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_enriched.py tests/test_generate_enriched.py
git commit -m "feat(SP-L): enriched pocket generation script (weights in, indices out)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 6: Orchestrator — one round (`optimize_loop.py`)

**Files:**
- Create: `scripts/optimize_loop.py`
- Test: `tests/test_optimize_loop.py`

**Interfaces:**
- Consumes: `enrich.passes_gate`, `enrich.compute_enrichment_weights`, `enrich.molecule_index_sets`, `synformer.dock` (`dock`, `prepare_target`, `ReceptorSpec`).
- Produces:
  - `read_candidates(path) -> list[dict]` — parse candidates.jsonl.
  - `gate_and_dedup(records, sa_max=4.0) -> list[dict]` — keep `passes_gate` records, dedup by smiles, preserve order.
  - `dock_budget(records, spec, dock_fn, budget, seed, max_workers=4) -> dict[str, float]` — dock the first `budget` gated candidates concurrently (ThreadPool; smina is subprocess-bound), returning `{smiles: score}` excluding nan.
  - `select_winners(scored: dict[str, float], k: int) -> list[str]` — k most-negative (strongest) scores.
  - `next_weights(winner_smiles, all_records, w_max=5.0) -> dict` — map winners+pool to index-sets via their records and call `compute_enrichment_weights`; return JSON-serialisable `{"bb": {...}, "tpl": {...}}`.

- [ ] **Step 1: Write the failing test** (dock stubbed — no smina)

```python
# tests/test_optimize_loop.py
from scripts.optimize_loop import (
    gate_and_dedup, dock_budget, select_winners, next_weights,
)


def _rec(smi, bb, tpl):
    return {"smiles": smi, "bb": bb, "tpl": tpl}


def test_gate_and_dedup_drops_invalid_and_dupes(monkeypatch):
    import scripts.optimize_loop as ol
    monkeypatch.setattr(ol, "passes_gate", lambda s, sa_max=4.0: s != "BAD")
    recs = [_rec("A", [1], [1]), _rec("A", [1], [1]), _rec("BAD", [2], [2]), _rec("C", [3], [3])]
    out = gate_and_dedup(recs)
    assert [r["smiles"] for r in out] == ["A", "C"]


def test_dock_budget_excludes_nan_and_respects_budget():
    recs = [_rec(s, [1], [1]) for s in ["A", "B", "C"]]
    scores = {"A": -7.0, "B": float("nan"), "C": -5.0}
    fn = lambda spec, smi, seed=0: scores[smi]
    got = dock_budget(recs, spec=None, dock_fn=fn, budget=3, seed=1, max_workers=2)
    assert got == {"A": -7.0, "C": -5.0}  # nan dropped
    got2 = dock_budget(recs, spec=None, dock_fn=fn, budget=1, seed=1, max_workers=2)
    assert set(got2) == {"A"}  # budget honoured (first gated candidate)


def test_select_winners_takes_most_negative():
    assert select_winners({"A": -7.0, "B": -3.0, "C": -9.0}, 2) == ["C", "A"]


def test_next_weights_promotes_winner_building_blocks():
    recs = [_rec("A", [1], [1]), _rec("B", [1], [1]), _rec("C", [9], [9]), _rec("D", [9], [9])]
    w = next_weights(["A", "B"], recs, w_max=5.0)
    assert w["bb"].get("1", 1.0) > 1.0
    assert "9" not in w["bb"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_optimize_loop.py -q`
Expected: FAIL with `ModuleNotFoundError: scripts.optimize_loop`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/optimize_loop.py (helpers only in this task; CLI added in Task 7)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_optimize_loop.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/optimize_loop.py tests/test_optimize_loop.py
git commit -m "feat(SP-L): loop helpers (gate/dedup, parallel dock budget, winner select, next weights)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 7: Orchestrator — multi-round, two arms, resumable CLI

**Files:**
- Modify: `scripts/optimize_loop.py`
- Test: `tests/test_optimize_loop.py`

**Interfaces:**
- Consumes: Task 6 helpers; `synformer.dock` (`dock`, `prepare_target`); `scripts.generate_enriched` (via subprocess).
- Produces:
  - `run_generation(ckpt, target, weights_path, n, seed, out_path, python=".venv-train/bin/python")` — subprocess call to `generate_enriched`; raises on non-zero exit.
  - `round_dir(base, target, arm, r) -> pathlib.Path` and `is_round_done(round_dir) -> bool` (dock_scores.csv present & non-empty).
  - `run_arm(...)` — R rounds for one (target, arm): generate → gate/dedup → dock B → select k → write `dock_scores.csv`, `weights_next.json`, `candidates.jsonl`; resumable (skip completed rounds). Uniform arm passes `weights="NONE"` every round.
  - CLI: `optimize_loop --targets <json> --ckpt <p> --arms enrich,uniform --rounds 3 --budget 150 --n 1000 --k 30 --final-m 10 --seed 42 --out-dir data/dock/sp_l --limit-targets N`.
  - Emits `data/dock/sp_l/loop_summary.csv` (`target,arm,round,n_gated,n_docked,best,top10_mean,scaffold_div`) and per-(target,arm) `final_topM.smi`.

- [ ] **Step 1: Write the failing test** (resumability + arm weights, generation & dock stubbed)

```python
# append to tests/test_optimize_loop.py
import json, pathlib
import scripts.optimize_loop as ol


def test_is_round_done_requires_nonempty_scores(tmp_path):
    d = tmp_path / "r0"; d.mkdir()
    assert ol.is_round_done(d) is False
    (d / "dock_scores.csv").write_text("")
    assert ol.is_round_done(d) is False
    (d / "dock_scores.csv").write_text("smiles,score\nA,-7\n")
    assert ol.is_round_done(d) is True


def test_run_arm_resumes_and_uniform_uses_no_weights(tmp_path, monkeypatch):
    # stub generation: write a fixed candidate file; record the weights arg seen each round
    seen_weights = []
    def fake_gen(ckpt, target, weights_path, n, seed, out_path, python=None):
        seen_weights.append(pathlib.Path(weights_path).name if weights_path not in (None, "NONE") else "NONE")
        pathlib.Path(out_path).write_text(
            "\n".join(json.dumps({"smiles": s, "bb": [1], "tpl": [1]}) for s in ["A", "B", "C"]))
    monkeypatch.setattr(ol, "run_generation", fake_gen)
    monkeypatch.setattr(ol, "passes_gate", lambda s, sa_max=4.0: True)
    monkeypatch.setattr(ol, "dock", lambda spec, smi, seed=0: {"A": -9.0, "B": -5.0, "C": -3.0}[smi])
    ol.run_arm(ckpt="x", target="T", arm="uniform", spec=None, rounds=2, budget=3, n=3, k=1,
               seed=1, out_dir=tmp_path)
    assert seen_weights == ["NONE", "NONE"]  # uniform never enriches
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_optimize_loop.py -q`
Expected: FAIL (`is_round_done` / `run_arm` not defined).

- [ ] **Step 3: Write minimal implementation** — append to `scripts/optimize_loop.py`:

```python
import csv
import subprocess

import click
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from synformer.dock.dock import dock
from synformer.dock.receptor import prepare_target


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
            import pandas as pd
            df = pd.read_csv(scores_csv)
            scored = dict(zip(df.smiles, df.score))
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
            nw = next_weights(winners, recs)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_optimize_loop.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Dry-run on the box** — 1 target, tiny budget, both arms, confirm resumability:

```bash
cd ~/pw && export SMINA="$(pwd)/smina.static"
.venv/bin/python -m scripts.optimize_loop --ckpt "<SP-C ckpt>" \
  --targets data/dock/powered_targets.json --limit-targets 1 \
  --rounds 2 --budget 6 --n 12 --k 3 --final-m 3 --out-dir /tmp/sp_l_dry
# re-run the same command: rounds should skip (is_round_done) and not re-dock.
```

- [ ] **Step 6: Commit**

```bash
git add scripts/optimize_loop.py tests/test_optimize_loop.py
git commit -m "feat(SP-L): multi-round two-arm resumable loop orchestrator + summary

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 8: Pilot run + specificity readout

**Files:**
- Create: `scripts/sp_l_specificity.py` (thin driver that assembles each arm's final top-M into per-target candidate files and invokes the existing all-pairs harness)
- Modify: `docs/SP_L_RESULTS.md` (created in Task 10; numbers appended here)

**Interfaces:**
- Consumes: `data/dock/sp_l/<target>/<arm>/final_topM.smi`; `scripts.powered_run`; `scripts.powered_analyze`.
- Produces: `data/dock/sp_l/candidates_<arm>/<target>.txt` (top-M per target, powered_run candidate format), all-pairs matrices `dock_scores_sp_l_<arm>.csv`, and the normalized-delta comparison enrich vs uniform (+ SP-C reference).

- [ ] **Step 1: Run the pilot loop** (detached, ~19h; GPU 0 free):

```bash
cd ~/pw && export SMINA="$(pwd)/smina.static"
CKPT="logs/pocket/2607091019-32f2194@powered-specificity/2026_07_09__10_19_15/checkpoints/epoch=1-step=2255.ckpt"
# pilot targets = first 5 in powered_targets.json (O43570,P06537,P10721,P02753,P0C559)
setsid nohup .venv/bin/python -m scripts.optimize_loop --ckpt "$CKPT" \
  --targets data/dock/powered_targets.json --limit-targets 5 \
  --arms enrich,uniform --rounds 3 --budget 150 --n 1000 --k 30 --final-m 10 \
  --out-dir data/dock/sp_l > logs/sp_l_loop.log 2>&1 </dev/null &
# monitor: grep -E "final top|written" logs/sp_l_loop.log ; cut -d" " -f1-3 /proc/loadavg
```

- [ ] **Step 2: Build per-arm candidate files** from `final_topM.smi` into the `candidates_<arm>/` layout `powered_run` expects (one SMILES per line, `<target_id>.txt`). Write `scripts/sp_l_specificity.py` `assemble_candidates(out_dir, arm)` for this; unit-test it on a tmp dir with 2 fake targets.

- [ ] **Step 3: Run the all-pairs 5×5 matrix per arm** (reuse `powered_run` with `--candidates-dir data/dock/sp_l/candidates_<arm>`, `--limit-targets 5`, own+mismatch crystal arm; `SMINA` exported):

```bash
for ARM in enrich uniform; do
  setsid nohup .venv/bin/python -m scripts.powered_run \
    --targets data/dock/powered_targets.json \
    --candidates-dir data/dock/sp_l/candidates_$ARM \
    --scores data/dock/dock_scores_sp_l_$ARM.csv \
    --matrix-out data/dock/matrix_sp_l_$ARM.json \
    --n-candidates 10 --n-refs 0 --top-m 10 --limit-targets 5 \
    --work-dir boltz_out/sp_l_$ARM > logs/sp_l_matrix_$ARM.log 2>&1 </dev/null &
done
```

- [ ] **Step 4: Compute normalized delta per arm** with `powered_analyze` and record enrich vs uniform (+ existing SP-C −0.714 reference). Add `sp_l_specificity.py compare()` to print both deltas, the paired difference (enrich − uniform per target), and bootstrap CI.

- [ ] **Step 5: Commit** code + generated JSON summaries (not the large CSVs, which live on the share).

```bash
git add scripts/sp_l_specificity.py tests/test_sp_l_specificity.py
git commit -m "feat(SP-L): specificity readout driver (assemble top-M, all-pairs, enrich vs uniform delta)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 9: Boltz-2 out-of-loop validation

**Files:**
- Modify: `docs/SP_L_RESULTS.md` (Boltz corroboration section)

**Interfaces:**
- Consumes: `data/dock/sp_l/<target>/<arm>/final_topM.smi`; `scripts.boltz_matrix` / `scripts.boltz_controls` (`.venv-boltz`, `--accelerator gpu --no-kernels`).
- Produces: Boltz affinity for the final top-M (enrich arm) per target vs its own pocket; corroboration verdict vs docking.

- [ ] **Step 1: Prepare Boltz inputs** for each pilot target's enrich-arm final top-M (reuse `boltz_matrix_prepare` conventions: target sequence + top-M SMILES). Only the enrich arm's top-M (the arm we claim wins) needs the honesty check; note this scope in the log.

- [ ] **Step 2: Run Boltz** (`.venv-boltz`, GPU 0, detached):

```bash
cd ~/pw && setsid env CUDA_VISIBLE_DEVICES=0 nohup .venv-boltz/bin/python -m scripts.boltz_matrix \
  --inputs <sp_l_boltz_inputs.json> --out data/dock/sp_l_boltz.csv \
  --accelerator gpu --no-kernels > logs/sp_l_boltz.log 2>&1 </dev/null &
```

- [ ] **Step 3: Corroboration check** — for the enrich arm, does Boltz affinity rank the loop's top-M above a random-REAL baseline (reuse `boltz_controls_analyze` AUROC pattern)? Record whether co-folding agrees with the docking win or contradicts it (method-dependent), mirroring `BOLTZ_VALIDATION_RESULTS.md`.

- [ ] **Step 4: Commit** any new analysis code + the verdict into the results doc (Task 10).

---

### Task 10: Results doc + ledger

**Files:**
- Create: `docs/SP_L_RESULTS.md`
- Modify: `.superpowers/sdd/progress.md`

- [ ] **Step 1: Write `docs/SP_L_RESULTS.md`** — question, method (frozen model + BB/template enrichment, two budget-matched arms), the round-over-round top-M curve (secondary), the primary normalized-delta comparison (enrich vs uniform vs SP-C reference) with CIs, the Boltz corroboration verdict, diversity trajectory, caveats (5-target CIs wide; docking is shape-fit; enrichment scoped to retrieved BBs), and exact reproduce commands.

- [ ] **Step 2: Append the honest verdict** — one of: (a) enrichment beats budget-matched uniform on specificity (loop confers targeting selection can't get from more draws); (b) tie/null (enrichment = more draws); (c) enrichment improves affinity but not specificity (promiscuity). State which, with numbers.

- [ ] **Step 3: Update the SDD ledger** with a `## PLAN: SP-L enrichment-selection loop` block summarizing tasks, decisions, and the verdict.

- [ ] **Step 4: Commit**

```bash
git add docs/SP_L_RESULTS.md .superpowers/sdd/progress.md
git commit -m "docs(SP-L): enrichment-selection loop results + verdict + ledger

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

## Notes for the executor

- **Sync-back:** this runs on the box (a git mirror). After the branch is done, the commits must be
  synced to the laptop (git source of truth) — flag it, don't assume the box is authoritative.
- **The pure-logic tasks (1-3, 6 helpers) carry the real coverage.** GPU/dock/Boltz steps are ops:
  run detached, monitor by log, and never re-dock a completed round (idempotency is tested).
- **If `sample_pocket`'s return shape differs** from `(info, stacks)`, adapt only the call site in
  `generate_enriched.main`; `stacks_to_records` stays pure and tested.
- **Budget honesty:** both arms dock `rounds × budget` per target — keep them equal. If the run is cut
  short, cut both arms symmetrically.
