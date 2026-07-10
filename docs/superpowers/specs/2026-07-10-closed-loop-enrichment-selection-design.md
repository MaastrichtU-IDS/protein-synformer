# SP-L: Enrichment-Selection Loop — Design

**Date:** 2026-07-10 · **Sub-project:** Priority 1 (closed-loop generation) from `SESSION_HANDOFF.md`
· **Depends on:** the frozen SP-C pocket-conditioned checkpoint and the powered specificity harness.

## 1. Motivation & scientific claim

The project established **two nulls**: conditioning the SynFormer generator on target information —
sequence (SP2) or 3D pocket (SP-C) — does **not** confer target-specificity. What *did* produce a
modest, method-dependent specificity signal was **selecting** docked candidates against the 3D pocket.

This sub-project operationalizes that finding as a **closed loop that selects better from the frozen
SP-C model**, and asks the sharper question:

> Given that static pocket conditioning was a null on targeting, can iteratively **enriching**
> generation toward this round's docking-winners produce molecules that are more **target-specific** —
> not merely better binders everywhere?

The generator weights are **never updated** (this is a selection/importance loop, not reward
fine-tuning). The novelty over the one-shot SP-C selection already done is the round-over-round
**building-block / reaction-template enrichment** that re-biases the frozen model's sampling toward
synthetic motifs found in docking-winners.

### Readouts

- **Primary (science):** own-vs-mismatch **normalized delta** on an all-pairs docking matrix (per-pocket
  z-normalized; more negative = more specific), reusing `powered_run` + `powered_analyze`.
  **Win** = the loop's final top-M delta is significantly more negative than the SP-C one-shot baseline
  (crystal −0.714, from `POCKET_CONDITIONED_RESULTS.md`).
- **Secondary (capability):** top-M own-pocket docking score improves round-over-round (a convergence
  curve). This is supporting evidence, not the verdict.

A loop that only makes molecules bind *everything* better (promiscuity) will **not** move the primary
delta — that is exactly the confound the all-pairs mismatch matrix controls for.

## 2. The loop

Per target, `R` rounds:

```
pool_0  ← generate N candidates from the frozen SP-C model (uniform enrichment weights)
for r in 1..R:
    dock a budget B of pool_{r-1} against the target's OWN pocket (parallel driver)
    winners_r ← top-k by drug-like-guarded docking score
    weights_r ← enrichment(winners_r' synthetic routes):
                  per-building-block and per-reaction-template frequency ratio
                  (winners vs the full docked pool), clipped/normalized
    pool_r  ← generate N candidates from the frozen model, retrieval REWEIGHTED by weights_r
final_topM ← best-docked molecules across all rounds (drug-like-guarded)
```

Defaults (config knobs): pool size `N ≈ 1000`/round (generation is cheap, so over-generate and let the
drug-like gate + dedup thin it before docking `B` of them), winners `k = 30` (~top 10% of `B`),
final `M = 10`. Generation is cheap on GPU; **docking is the only expensive step**, so the budget `B` is
spent solely on evaluation. Each round's pool is regenerated fresh from the frozen model under the new enrichment
weights (not mutated from the previous pool), preserving the model's chemistry while shifting mass toward
winner motifs.

### Enrichment weights

For each round, from the winners' `Stack`s (SynFormer synthetic routes):

- **Per-building-block weight:** `w_bb(i) = clip( freq_winners(i) / (freq_pool(i) + eps), 1, w_max )`
  for building-block fpindex-index `i`; `1.0` for unseen indices (no down-weighting — enrichment only
  *promotes*).
- **Per-template weight:** analogous ratio over reaction-template indices used in winner routes.

`eps`, `w_max` (e.g. 5.0) are config. Weights of `1.0` everywhere ⇒ identical to the frozen model
(round 0 / degenerate fallback).

### The reweight hook (frozen-model, no weight update)

At `synformer/models/synformer.py:366–381`, generation samples:
- the **reaction** via `softmax(reaction_logits / T_rxn)` — over **all** templates (global), and
- the **reactant (building block)** via `softmax(fp_scores / T_reactant)` over the **top-k retrieved**
  BBs for this step, where `pred.retrieved_reactants.indices` gives each candidate's fpindex index.

Enrichment adds `log(w)` to the pre-softmax scores of both channels before the multinomial — one
convention so a weight `w` cleanly multiplies the resulting sampling probability by `w`, independent of
the temperatures:
- reactant: `fp_scores ← T_reactant · log(fp_scores) + log(w_bb[indices])` (or equivalently bias the
  logits the softmax consumes); **scoped to the retrieved set** — see caveat.
- reaction: `reaction_logits ← reaction_logits + log(w_template)` — global, no scope limit.

**Scope caveat (important).** Reactant reweighting can only promote a winner building block that is
**already among the top-k retrieved** for that step; it cannot pull in a BB the retrieval missed. So this
is "bias *selection among retrieved* BBs," not "bias retrieval." Two consequences the implementation must
handle: (1) check the generation-time retrieval `k` — if it is small, enrichment has little room, and
widening `k` during the loop should be considered; (2) template enrichment (global) does **not** have
this limitation and may carry most of the enrichment signal.

Weights are passed through the existing `**options` channel of `predict` / `generate_without_stack`
(default `None` ⇒ current behavior, so every existing caller is unaffected). No parameters are modified;
`code` (the pocket conditioning) is untouched, so the log-likelihood/conditioning path is unchanged.

## 3. Anti-hacking guards

Optimizing against a docking proxy risks producing **smina-hackers** (malformed/strained molecules smina
scores well but that are not real binders — this project's finding #1). Because we *select* rather than
gradient-optimize, the risk is lower, but the enrichment still amplifies whatever wins. Guards:

- **Winner gate:** a molecule is eligible to be a winner only if RDKit-valid **and** SA ≤ 4
  **and** drug-like (reuse `dock_prepare.is_drug_like_ligand` heavy-atom/element criteria).
- **Docking hygiene:** `nan` docks (embed failure / timeout) are excluded from winners and logged; they
  never enter the enrichment statistics.
- **Diversity monitoring:** per-round scaffold diversity and uniqueness are computed and logged. If
  enrichment collapses diversity, that is reported as a result (loop narrows chemistry), not hidden.

## 4. Independent validation gate (Boltz-2)

smina is a rigid shape-fit proxy that this project already showed can **disagree with co-folding**
(`BOLTZ_VALIDATION_RESULTS.md`). So the loop's **final selected top-M per target** is re-scored with
**Boltz-2** (`.venv-boltz`, reusing `scripts/boltz_matrix.py` / `boltz_controls.py`) as an *independent*
readout — **not inside the loop** (too slow). A loop win that Boltz corroborates is a strong result; a
win Boltz contradicts is reported as method-dependent, exactly as the docking-selection specificity was.
Boltz runs on the final top-M only (tens of cells), so it is cheap relative to the loop.

## 5. Scope & docking budget

Docking throughput on the box is ~240 docks/hr (smina ~7 cores/dock ⇒ ~4 concurrent, ~60 s each), so
docking is the binding constraint. This spec covers a **pilot**, not the full 20-target study.

- **Pilot targets (5):** the original docking-selection set, which already have SP-C baseline candidates
  and known-ligand docking —
  `O43570_WT` (CA12/1JD0/AZM), `P06537_WT` (GR/3MNP/DEX), `P10721_WT` (KIT/1T46/STI),
  `P02753_WT` (RBP4/1BRP/RTL), `P0C559_WT` (gyraseB/6Y8O/NOV).
- **Rounds:** `R = 3`. **Budget:** `B ≈ 150` docks/round/target/arm.
- **Two docking-budget-matched arms (the causal comparison):**
  - **Enrichment arm** — the loop of §2 (`enrich_weights` updated each round from winners).
  - **Uniform control arm** — identical structure and identical total docking budget, but
    `enrich_weights = None` every round (plain re-sampling from the frozen model). This is the
    within-round "select more but don't enrich" control.
  Both arms dock `R×B = 450`/target; total loop docking ≈ 5×2×450 = 4500 docks.
- **Specificity matrix:** all-pairs 5×5 over each arm's final top-M (M=10) → ≈ 2×5×10×5 = 500 matrix
  docks (own-pocket cells reuse loop docks). Total ≈ 19 h wall on the idle box.
- **Full 20-target scale-up is an explicit compute follow-on**, gated on the pilot showing signal.

**Primary comparison — enrichment arm vs uniform control arm, at equal docking budget.** This isolates
the loop's *enrichment* contribution from the trivial "sampled and docked more draws" effect: both arms
draw and dock the same total, so any specificity difference is attributable to enrichment, not budget.
A *secondary*, non-causal reference is the existing **SP-C one-shot** ~150-pool top-M
(`data/dock/candidates_pocket/`) — reported for continuity with `POCKET_CONDITIONED_RESULTS.md`
(baseline −0.714), but **not** the headline comparison because it is not budget-matched.

Note the enrichment selects on the **own-pocket diagonal only** (it optimizes affinity to the target),
so it may yield promiscuous better-binders and a specificity null. Either outcome is a valid result — and
the budget-matched control is what makes either interpretable.

## 6. Components

| Piece | New / reuse | Notes |
|---|---|---|
| `synformer/models/synformer.py` — enrichment reweight passthrough | **small new** | `options["enrich_weights"]` (per-BB, per-template); default `None` ⇒ unchanged |
| `synformer/molopt/enrich.py` — mine `Stack`s → enrichment weights | **new (TDD)** | frequency-ratio math; pure, unit-testable |
| `scripts/optimize_loop.py` — orchestrator | **new (TDD)** | generate→dock→enrich→iterate; runs both the enrichment arm and the budget-matched uniform arm; idempotent/resumable per-round artifacts |
| `sample_helpers.sample_pocket` / `build_pocket_feat` | reuse | pass enrichment weights through to `generate_without_stack` |
| `synformer.dock` + `pocket_dock_driver.sh` (+ `SMINA` env) | reuse | docking; parallel driver already caps 4 concurrent |
| `powered_run` (all-pairs) + `powered_analyze` | reuse | specificity matrix + normalized delta + bootstrap CIs |
| `boltz_matrix` / `boltz_controls` | reuse | final-top-M independent validation |
| `dock_prepare.is_drug_like_ligand`, SA scorer | reuse | winner gate |

Artifacts (docking is expensive → durable, resumable) live on the NFS share under
`data/dock/sp_l/<target>/<arm>/round_<r>/…` (`arm` ∈ {`enrich`, `uniform`}) with `candidates.smi`,
`dock_scores.csv`, `weights.json`, and a top-level `loop_summary.csv` (per-arm, per-round top-M score +
diversity).

## 7. Error handling

- **nan docks** (embed/timeout): excluded from winners and from enrichment stats; counted and logged.
- **Empty/degenerate winners** (all nan, or <k eligible): enrichment falls back to uniform weights
  (`w=1`), warns, and the round proceeds as a plain re-sample.
- **Idempotency:** existing round artifacts are detected and **not re-docked** (mirrors
  `dock_select`/`powered_run` restart logic). Partial rounds resume from the last completed dock set.
- **Determinism:** per-round RNG seeded from `(base_seed, target, round)`; ETKDG conformer seed stays
  fixed (as in `synformer.dock.dock`) so a given SMILES always docks identically.

## 8. Testing (TDD)

- `enrich.py`: frequency-ratio correctness on a hand-built pool/winners set; clip/normalize; empty
  winners → uniform; unseen indices → weight 1.0.
- reweight hook: on a toy fpindex + toy weights, verify (a) sampling mass shifts toward an up-weighted
  BB **that is present in the retrieved top-k**; (b) an up-weighted BB **absent** from the retrieved set
  has **no effect** (documents the selection-not-retrieval boundary); (c) template up-weighting shifts
  reaction choice globally; (d) `log(w)` convention gives the same probability-multiplier in both
  channels independent of temperature; (e) `enrich_weights=None` reproduces baseline sampling exactly
  (regression guard for existing callers).
- orchestrator: dry-run round transition (generate→dock stub→enrich→next weights); restart-skip
  (existing round artifacts not re-docked); nan handling; drug-like gate applied.
- Regression: full existing suite (106 pass on `main`) stays green — the `options` default guarantees
  unchanged behavior for `dock_prepare`/`powered_run`/`sample_helpers`.

## 9. Explicit non-goals (YAGNI)

- **No generator weight updates** (no REINVENT/DPO/reward-FT) — deferred; a different scientific claim.
- **No surrogate scorer** — docking is the in-loop scorer; Boltz is out-of-loop validation only.
- **No fragment/scaffold-seeding** mechanism — enrichment is building-block/template only.
- **No full 20-target run** in this spec — pilot first; scale-up is a compute follow-on.
- **No consensus-in-the-loop scorer** — deferred.

## 10. Reproduce (target commands, filled in during implementation)

- Loop: `python -m scripts.optimize_loop --targets <5> --rounds 3 --budget 300 --ckpt <SP-C> …`
  (`SMINA=$(pwd)/smina.static`, detached via `setsid nohup … </dev/null &`).
- Specificity: `powered_run` all-pairs over final top-M → `powered_analyze` → normalized delta vs SP-C
  baseline.
- Validation: `boltz_matrix` on final top-M → corroboration check.
