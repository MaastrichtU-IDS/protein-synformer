# Powered Target-Specificity Study — Results & Verdict

**Date:** 2026-07-09 · **Branch:** `powered-specificity` · **Spec/plan:** `docs/superpowers/{specs,plans}/2026-07-08-powered-target-specificity*`

## Bottom line

The docking-selection POC found a target-specificity signal on **5 hand-picked pristine targets**
(pocket-normalized own<offdiag, win 5/5, delta −1.74), which an independent co-folding check (Boltz-2,
N=5) did **not** corroborate. This study powered that to **N=20 test-split targets** with bootstrap
CIs, added an **AlphaFold-structure docking arm** (crystal vs predicted pocket) as an artifact test,
and re-checked Boltz competence at scale. Findings:

1. **Docking specificity survives at N=20 — but modest.** Pocket-normalized own−offdiag delta
   **−0.62, 95% CI [−1.02, −0.25]** (excludes 0 → statistically real), win-rate **0.70 [0.50, 0.90]**.
   Real, but far weaker than the pristine-5 (−1.74, 5/5) — the pristine targets overstated it.
2. **It is NOT an experimental-crystal artifact.** Docking into **AlphaFold-predicted** pockets gives
   a comparable signal (delta **−0.45 [−0.96, +0.06]**), and the **crystal−AF paired difference is
   −0.13, CI [−0.61, +0.34] — includes 0** (no significant difference). So the effect is a property of
   **rigid-pocket shape complementarity of any well-formed structure**, not the exact holo crystal.
3. **Boltz stays competent at scale**, so its co-folding null remains informative: known-vs-random
   AUROC **0.845 [0.789, 0.898]** across 20 targets (down from the pristine-5's 0.95, still far above
   chance).

**Synthesis:** rigid-receptor docking (crystal *or* predicted) shows a **modest, real** own-pocket
preference; flexible co-folding (Boltz-2, competent here) shows **none**. The target-specificity of the
generated molecules is therefore **weak and method-dependent** — a rigid-pocket shape-fit effect that
does not transfer to a flexibly-folded complex. Docking selection surfaces molecules that fit a rigid
pocket somewhat preferentially; it does not deliver strong, method-robust target selectivity.

Everything ran on an NVIDIA A40 box (docking on 32 CPU cores, Boltz on GPU); code + data are on branch
`powered-specificity`.

---

## Setup

- **20 targets** (superset of the original 5), test-split, single drug-like holo, ≥10 known ligands
  each, 19 protein families. ~150 SP2-`masked` candidates/target. `data/dock/powered_targets.json`.
- **Full N×N mismatch matrix** (not a panel): every target's top-10 selected candidates docked into
  **every** target's pocket, so each pocket column is full and the own-pocket term is per-column
  z-normalized cleanly — identical methodology to the trusted N=5. Crystal arm: **20×20 complete**.
- **AlphaFold arm**: each pocket re-rendered from its `AF-<acc>-F1-model_v6.pdb`, superposed onto the
  holo crystal (single ligand-binding chain), crystal ligand transplanted as the autobox; top-10 hits
  docked into the AF pockets. A quality gate (anchor count + CA-RMSD) excluded **2/20 AF pockets**
  (`P48729`: AF↔crystal alignment too weak; `Q92847`: superposition edge case) → **AF arm 20 sources ×
  18 pockets**, nan-aware.
- **Analysis**: `scripts/powered_analyze.py` — per-column z-normalized own−offdiag delta + win-rate,
  **bootstrap 95% CIs** over targets, the crystal−AF paired difference, and Boltz known-vs-random AUROC.
- **Boltz** discrimination control (known vs random, cap 10/class/target) across all 20, on the A40.

## Results

| arm | N | normalized delta [95% CI] | win-rate [95% CI] | reading |
|---|---|---|---|---|
| **crystal docking** | 20 | **−0.62 [−1.02, −0.25]** | 0.70 [0.50, 0.90] | own<offdiag, CI excludes 0 → **real, modest** |
| **AlphaFold docking** | 18 | −0.45 [−0.96, **+0.06**] | 0.61 [0.39, 0.83] | same direction/size; CI includes 0 (N=18 underpowered) |
| **crystal − AF (paired)** | 18 | **−0.13 [−0.61, +0.34]** | — | includes 0 → **no significant difference** |
| **Boltz known-vs-random AUROC** | 20 | 0.845 [0.789, 0.898] | — | ≫0.5 → **competent at scale** |

(delta < 0 = a target's selected molecules dock better into its own pocket than into mismatched
pockets, after removing per-pocket dockability; lower = more specific.)

## Interpretation

- **The N=5 win-5/5 was a pristine-target overstatement.** At N=20 the effect is real (crystal CI
  excludes 0) but modest (delta −0.62 vs −1.74; win 70% not 100%). Honest scaling shrank it.
- **Structure origin doesn't matter; rigidity does.** AF-predicted pockets reproduce the crystal
  signal (crystal−AF ≈ 0). So the docking specificity is not an artifact of the *experimental* pocket —
  it is what rigid-receptor docking sees for a well-formed pocket of *either* origin. Combined with the
  N=5 co-folding result (no specificity) from a method now confirmed competent at scale (AUROC 0.85),
  the thing that erases the signal is **conformational flexibility / co-folding**, not crystal-vs-AF.
- **So target-specificity is weak and method-dependent.** Docking selection reliably finds
  synthesizable molecules that *fit a rigid pocket* somewhat preferentially, but that preference is
  small and not corroborated by flexible co-folding — it should not be read as robust molecular target
  selectivity.

## Caveats

- **Still modest power:** N=20 (AF 18); the crystal win-rate CI lower bound touches 0.5, and the AF
  delta CI barely includes 0 — directionally consistent but not a large, unambiguous effect.
- **Docking is a proxy** (smina empirical scoring), not experimental affinity; the *relative*,
  pocket-normalized framing mitigates but does not remove this.
- **AF arm excludes 2 pockets** (gate-rejected); AF own-pocket terms exist for 18/20 targets.
- **Single seed**; per-target candidate pools of ~140–160; the analysis normalization is per-column z.
- **Boltz-2 generated-molecule caveat** still applies to reading its co-folding null as ground truth,
  though the AUROC-0.85 competence control substantially blunts the "Boltz just can't tell" objection.

## What this means

The powered study **refines** the project's arc: the docking target-specificity is *real but small*,
is a *rigid-pocket shape-fit* property (crystal or AlphaFold alike), and *does not survive flexible
co-folding* — so the deployable claim is "docking selection enriches for pocket-fitting, synthesizable
molecules," not "target-specific binders." The natural next steps remain: more targets/seeds for
tighter CIs, a co-folding tool validated on de-novo molecules, and ultimately experimental binding
data on the top hits.

## Artifacts
- Scores: `data/dock/dock_scores.csv` (crystal 20×20), `data/dock/dock_scores_af.csv` (AF 20×18),
  `data/boltz/boltz_controls_scores.csv` (discrimination). Summary: `data/dock/powered_specificity_summary.csv`.
- Reproduce: `.venv/bin/python -m scripts.powered_analyze`. Targets: `data/dock/powered_targets.json`.
- Code: `scripts/powered_targets.py`, `synformer/dock/af_receptor.py`, `scripts/powered_run.py`
  (`--source-shard`/`--sources`/`--work-dir` for parallel docking), `scripts/powered_analyze.py`.
