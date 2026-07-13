# Tier-2 Calibration: docking selectivity tracks MEASURED selectivity — weakly, and only for kinases

**Date:** 2026-07-13 · Branch `powered-specificity` · Proposal:
`docs/SPECIFICITY_CALIBRATION_PROPOSAL.md` · Scripts: `scripts/tier2_{fetch,pairs,analyze}.py`,
`tier2_dock_driver.sh` · Follows [TIER1_CALIBRATION_RESULTS](TIER1_CALIBRATION_RESULTS.md)

## Question (the decisive calibration)

Does the docking own-vs-mismatch selectivity metric track **measured** selectivity? For every compound
with ChEMBL pChEMBL on a pair of targets (A, B), correlate **measured** ΔpChEMBL(A−B) against **docked**
Δselectivity (z_A − z_B, z per pocket over the common docking set). Spearman overall, **within-family
(paralog — the valuable case)**, and cross-family.

## Data

ChEMBL measured pChEMBL (Ki/Kd/IC50/EC50) for the 6 docking-working panel targets: KIT 3183, JAK3 5784,
CDK5 720, 5-HT1A 5749, 5-HT2A 5432, A1R 5133 compounds. **1,732** compounds measured on ≥2 targets. Docked
**460** (all 260 within-kinase-pair compounds + 200 within-GPCR-pair) into all 6 pockets (2,760 docks) →
**530 (compound, A, B) triples**: 320 within-kinase, 201 within-GPCR, 9 cross-family.

## Result — real but weak, and kinase-specific

| stratum | n triples | Spearman ρ (docked vs measured; + = tracks) | 95% CI (compound-clustered bootstrap) |
|---|---|---|---|
| **within-kinase (paralog)** | 320 | **+0.245** | **[+0.13, +0.35] — excludes 0** |
| within-GPCR (subtype) | 201 | +0.052 | [−0.09, +0.20] ns |
| cross-family | 9 | +0.14 | (too few) |
| ALL | 530 | +0.196 | [+0.11, +0.28] sig |

The within-kinase signal is **robust** (survives a compound-clustered bootstrap, not just triple-level).

**Per-pair decomposition** (guards against one dominant pair carrying a stratum):

| pair | family | n (all unique compounds) | ρ |
|---|---|---|---|
| KIT/JAK3 | kinase | 101 | **+0.344** |
| KIT/CDK5 | kinase | 39 | **+0.315** |
| JAK3/CDK5 | kinase | 180 | **+0.173** |
| 5HT1A/5HT2A | GPCR | 200 | +0.048 |

**All three kinase pairs lean positive and consistent** (ρ 0.17–0.34) — and the *largest* pair (JAK3/CDK5,
n=180) is the *weakest*, so the signal is not a single-pair artifact; it generalizes across the kinase
paralog pairs. The GPCR stratum is effectively the **single** 5-HT1A/5-HT2A pair (the other aminergic
overlaps were too small to test), which is flat. Every pair is 100% unique compounds, so the ρ is not
manufactured by a few congeneric analog series.

## Interpretation

- **The docking-selection specificity is REAL** — it correlates with ground-truth measured selectivity,
  significantly, overall and for kinase paralogs. This is the first time any of the project's specificity
  signal has been anchored to measured data, and it **grounds the powered study's specificity** (that
  corpus was 39% kinases). On the within-family question, Tier-2 is a **more sensitive, measured-affinity
  test that resolves a weak paralog signal Tier-1 could not** — it is *consistent with* Tier-1's null
  (ρ≈0.25 is weak — exactly the regime an own-preference test on non-certified-selective binders would
  miss), not a reversal that makes Tier-1 "wrong."
- **But it is WEAK** — ρ ≈ 0.25 explains only ~6% of the rank variance in kinase selectivity. Docking
  captures a *minority* of what determines paralog selectivity. This is exactly why an independent scorer
  (Boltz, ρ-equivalent chance-level delta) does not corroborate at the molecule level: a weak signal is
  easily missed by a different method.
- **It is target-class-dependent** — **no detectable signal for aminergic GPCRs** (ρ 0.05, ns). Physically
  sensible: kinase paralog selectivity has steric/shape components in the ATP pocket that rigid docking
  partially captures; 5-HT/adenosine-receptor subtype selectivity is governed by subtler polar contacts
  and receptor conformational states that rigid smina misses.

## Verdict (resolves the calibration; per pre-committed rule)

Tier-2 lands between the proposal's outcomes: the metric is **not** noise (rescue side — within-kinase CI
excludes 0), but it is a **weak, class-specific affinity/shape signal, not a strong general selectivity
detector**. So:

> **Docking-selection specificity is a real but weak selectivity filter — usable for kinase paralogs,
> useless for aminergic GPCRs.** The project's positive is now *calibrated*: it was measuring something
> real (helped by a kinase-heavy corpus), at ρ≈0.25 strength, not the strong targeting the raw −0.6/−0.8
> delta implied.

## Implications / path forward

- **Selection with docking** is a legitimate but weak selectivity filter **for kinases only**. For general
  or strong selectivity it is insufficient.
- **A learned selectivity oracle is the credible path** — and the data assembled here (1,732 multi-target
  compounds with measured pChEMBL) is a ready **training set**. A grounded selection/DPO retry rewarded by
  a *measured-selectivity* oracle (not smina shape-fit) is the experiment that could finally confer real
  targeting — and Tier-2 tells us to prioritize kinases and be skeptical of GPCRs.
- **Allosteric/regulatory pockets** remain the untested lever for stronger paralog selectivity.
- **Better physics for GPCRs** (induced-fit/ensemble docking, or Boltz-2 affinity as the scorer) is needed
  before selection can touch GPCR subtype selectivity.

## Caveats

- ρ ≈ 0.25 is **weak** — a real but small effect; do not overstate it as "docking predicts selectivity."
- Within-GPCR ns could hide a very weak positive (CI upper +0.20); the honest read is "no usable signal."
- Cross-family is underpowered (n=9) — few compounds are tested on both a kinase and a GPCR.
- "GPCRs" here is really the **single** 5-HT1A/5-HT2A pair (other aminergic overlaps too small) — the
  GPCR null should not be generalized to all GPCRs on n=1 pair.
- Orthosteric crystal pocket, rigid smina. Per-(compound,target) activity is aggregated as **best
  pChEMBL**, which can bias toward targets with more assays; **median pChEMBL is preferable** and worth a
  rerun before any oracle build. Assay heterogeneity (Ki/IC50 mixed) adds noise that biases ρ *downward*
  (true signal may be a touch stronger).

## Bottom line

Calibrated against measured affinity, the docking specificity metric **is real but weak and kinase-
specific** (paralog ρ 0.245, sig; GPCR ns). The project's central positive is neither an artifact (Tier-1
worry) nor strong targeting (raw-delta impression) — it is a **weak, class-dependent selectivity signal**.
Real targeting needs a **learned selectivity oracle** (training data now in hand) and/or allosteric
targeting; docking-selection alone gets you weak kinase selectivity and nothing for GPCRs.
