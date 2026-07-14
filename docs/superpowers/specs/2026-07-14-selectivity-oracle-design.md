# SP-ORACLE (Stage A): Learned Selectivity Oracle — Design

> **SUPERSEDED (2026-07-14):** the advisor-mandated applicability-domain pre-check
> (`scripts/oracle_domain_check.py`) killed this before implementation — the generator's molecules sit at
> median Tanimoto ~0.27 to the nearest ChEMBL training compound (52–75% < 0.3), i.e. out of the oracle's
> domain, and unvalidatable there. No oracle was built. See `docs/SP_ORACLE_RESULTS.md`. The design below
> is retained for the record; the scaffold-split gate it specifies would have measured the wrong
> (in-distribution) quantity.

**Date:** 2026-07-14 · Sub-project: build a structure→affinity oracle on the assembled ChEMBL data and
**validate whether it predicts held-out measured selectivity better than docking** (Tier-2 baseline
ρ 0.245 within-kinase, ~0 within-GPCR). Stage A only; the generator retry (Stage B) is gated on this and
designed separately. · Depends on: `data/dock/tier2/raw/*.json` (ChEMBL cache), `data/dock/tier2/dock_scores.csv`
(docking baseline on the same compounds), the Tier-2 triples metric.

## 1. Goal & discipline

> Does a cheap learned oracle predict **held-out measured selectivity** (ΔpChEMBL between paralogs)
> **better than rigid docking** — enough to be worth using as a reward for a grounded generator retry?

The oracle must **prove itself against ground truth before** it is trusted as a reward — the same
discipline that ran through the whole calibration. Stage A is pure ML (no docking, no GPU), ~hours, and is
a **decision gate**: pass → Stage B (retry); fail → the oracle path is also weak, report and stop.

## 2. Data

- Source: `data/dock/tier2/raw/<target>.json` (already fetched) for the 6 panel targets KIT/JAK3/CDK5
  (kinase), 5-HT1A/5-HT2A/A1R (GPCR): 720–5784 compounds each with pChEMBL.
- **Aggregation: median pChEMBL preferred** per (compound, target) — max biases toward targets with more
  assays. The existing cache stored *best* pChEMBL. Median requires re-pulling all activities (slow through
  the proxy); the advisor's guidance is "don't rerun the world for it." So: **default to re-fetching median
  (resumable, background), but the cached best-pChEMBL is an acceptable documented fallback** if the
  re-fetch is impractical — the aggregation choice is a second-order label-noise effect, not the gate.
- Features: **Morgan fingerprint** (radius 2, 2048 bits) from `canonical_smiles`; optionally 4 physchem
  descriptors (MW, logP, HBD, TPSA) concatenated. One feature matrix; per-target label vectors with
  missing entries where a compound has no pChEMBL for that target.

## 3. Model

- **Per-target regressor** (one model per target), trained on that target's available (compound, pChEMBL)
  rows. sklearn `HistGradientBoostingRegressor` (default) — fast, strong on tabular/FP, handles the data
  scale; `RandomForestRegressor` as a documented fallback. No multi-output/masking needed — per-target
  models sidestep missing labels cleanly.
- Predicted selectivity for a compound between targets A,B: `pred_pChEMBL(A) − pred_pChEMBL(B)`.

## 4. Validation (load-bearing)

- **Scaffold split** (Bemis–Murcko generic scaffold via RDKit `MurckoScaffold`), a **single global 5-fold
  scaffold→fold assignment** shared across all targets, so a held-out compound is held out from *every*
  target model — required for honest held-out selectivity triples. Random split is prohibited (analog
  leakage inflates the oracle).
- For each test fold: train per-target models on the other folds, predict the test compounds.
- **Metrics on the pooled scaffold-held-out predictions:**
  1. Per-target affinity: Spearman ρ and R² (predicted vs measured pChEMBL).
  2. **The decisive metric — selectivity:** on the held-out (compound, A, B) triples (same construction as
     `tier2_analyze`), compute Spearman ρ between **measured** ΔpChEMBL and **oracle-predicted** Δ, split
     within-kinase / within-GPCR, with a compound-clustered bootstrap CI.
  3. **Head-to-head vs docking on the identical held-out triples:** for the same test compounds, also
     compute docking-selectivity ρ (from `dock_scores.csv`, the Tier-2 docks) — so oracle vs docking is
     compared on exactly the same molecules, not against the fixed 0.245 from a different set.

## 5. Decision gate (pre-committed)

- **PASS** (→ Stage B): oracle scaffold-held-out within-kinase selectivity ρ is **meaningfully higher than
  docking's on the same held-out triples** (bootstrap CIs separate, or oracle clearly higher), *and/or*
  the oracle shows real within-GPCR selectivity ρ (CI excludes 0) where docking was null. Either is a
  reason the oracle is a better reward than docking.
- **FAIL** (stop, report): oracle ≈ docking or worse on held-out selectivity → a learned oracle at this
  data scale doesn't beat rigid docking; the oracle-reward path is not worth the Stage-B build.
- **Honesty guards:** report affinity R² alongside selectivity ρ (a decent affinity model with poor
  selectivity ρ is the expected, informative middle case); a scaffold-held-out null is *inconclusive at
  this data scale*, not "learned oracles can't work" — n and the 6-target scope are the limits.

## 6. Components & interfaces

| file | responsibility |
|---|---|
| `scripts/oracle_data.py` | load ChEMBL cache → median pChEMBL per (compound,target); Morgan features; **scaffold_folds(smiles, k, seed)** (pure, TDD) |
| `scripts/oracle_train.py` | per-target scaffold-CV train/predict → held-out prediction table (compound,target,measured,pred) |
| `scripts/oracle_eval.py` | per-target R²/ρ; **selectivity ρ (oracle vs docking) on held-out triples** (reuse Tier-2 triple construction + compound-clustered bootstrap) |
| `docs/SP_ORACLE_RESULTS.md` | verdict + gate decision |

## 7. Testing (TDD)

- `scaffold_folds`: same scaffold never in two folds; every compound assigned; k folds roughly balanced.
- median aggregation: correct median per (compound,target) from multiple activities; missing handled.
- selectivity-Δ + triple construction: matches `tier2_analyze`'s convention (sign, per-pocket handling).
- Model train/predict smoke on a tiny synthetic set (fit→predict shape, deterministic seed).
- **Note:** the raw cache stored *best* pChEMBL per compound; §2 needs *median*. Task 1 re-fetches keeping
  all activities (or recomputes median) — cheap, and the fetch is already resumable.

## 8. Non-goals

- **Stage B (the generator retry) is NOT in this spec** — gated on the Stage-A verdict, separate design.
- No GNN/Chemprop (FP + GBM is the fast decisive baseline; GNN only if the gate passes and accuracy is the
  bottleneck).
- No expansion beyond the 6 panel targets (keeps the docking head-to-head exact).

## 9. Caveats

- 6 targets, kinase/aminergic-GPCR only; within-GPCR is effectively the single 5-HT1A/5-HT2A pair.
- ChEMBL assay heterogeneity (Ki/IC50/EC50 pooled) adds label noise (biases ρ down).
- The oracle predicts affinity for **these trained targets** — it is not pocket-general; Stage B's retry is
  therefore on the panel targets, and the oracle's generalization to the generator's novel chemotypes is
  itself a Stage-B risk (scaffold-held-out ρ here is the best available proxy for it).
