# Tier-1 Calibration Results: the specificity instrument does not measure selectivity

**Date:** 2026-07-13 · Branch `powered-specificity` · Proposal:
`docs/SPECIFICITY_CALIBRATION_PROPOSAL.md` · Scripts: `scripts/tier1_{prep,analyze}.py`,
`tier1_dock_driver.sh`

## Question

Does the own-vs-mismatch docking **z-delta** — the exact metric behind the project's central positive
("selection against the pocket confers specificity", delta −0.6/−0.8) — track **real** target selectivity?
Tier-1 test: dock **known actives**, **property-matched decoys** (MW/logP/HBD-matched, Tanimoto<0.35),
and the model's **generated candidates** (25 each) for 8 family-diverse targets (KIT/JAK3/CDK5 kinases;
5-HT1A/5-HT2A/A1R GPCRs; CA12; RAB9A) into a shared 8-pocket panel, and compare the metric across classes.

## Result — the metric does NOT separate real binders from decoys (falsification)

**Exact powered metric (top-M=10 per target, z-per-pocket-column delta; more negative = more own-preferring):**

| class | mean per-target delta (n=8) |
|---|---|
| **known actives** | **−0.05** |
| property-matched decoys | −0.43 |
| generated candidates | −0.44 |

Real known actives are the **least** own-preferring of the three — *not* more than matched decoys (actives
more-negative in only **3/8** targets; paired mean diff +0.38, ns at n=8). Decoys and candidates score as
*more* "specific" than actual known binders.

**Per-molecule z-delta (pooled, common per-pocket z):** actives −0.043, decoys +0.056, candidates −0.031.
**AUROC(z-delta ranks actives > decoys) = 0.541** (diff −0.099, CI [−0.24, +0.05], ns). The metric is
near-blind to whether a molecule is a real active or a matched decoy of that pocket. Candidates are
indistinguishable from actives (diff −0.012, ns).

## But smina is NOT noise — it has modest, family-dependent own-pocket AFFINITY signal

Raw **own-pocket** score does rank actives above matched decoys — **mean AUROC 0.66**, strongly
family-dependent:

| target | own-pocket actives vs decoys (kcal/mol) | AUROC |
|---|---|---|
| KIT (kinase) | −9.9 vs −8.9 | **0.80** |
| CDK5 (kinase) | −8.2 vs −7.7 | 0.74 |
| JAK3 (kinase) | −8.6 vs −8.0 | 0.73 |
| 5-HT1A (GPCR) | −7.6 vs −3.3 | 0.73 |
| 5-HT2A (GPCR) | −8.9 vs −8.6 | 0.65 |
| A1R (GPCR) | −8.4 vs −8.0 | 0.58 |
| RAB9A (GTPase) | −7.1 vs −6.9 | 0.56 |
| CA12 (lyase) | −6.4 vs −6.6 | 0.48 |

Generated candidates dock **comparably to real actives** on the own pocket (e.g. KIT −9.0 vs −9.9; A1R
−8.9 vs −8.4) — the generator makes genuinely good shape-fit binders.

## Interpretation

- **smina retains a real but modest AFFINITY signal** (own-pocket binder-vs-decoy AUROC 0.66) — strong for
  deep kinase/aminergic-GPCR pockets, at chance for shallow/polar CA12 and the GTPase.
- **The own-vs-mismatch normalization destroys most of it** (0.66 → 0.54) and yields a quantity that does
  **not** track molecular selectivity: real known binders are no more (point-estimate less) own-preferring
  than matched decoys.
- **Therefore the project's docking-selection "specificity" (−0.6/−0.8) is best read as a relative-
  normalization / shape-fit artifact, not molecular selectivity.** This *explains* Boltz's independent
  non-corroboration (own-vs-mismatch delta −0.04): both scorers agree the own-preference is not real
  selectivity. Docking-selection enriches **binders** (affinity + shape fit), not **selective** binders.

## Verdict (per pre-committed decision rule)

Tier-1's role is to **falsify**, and it does: the specificity metric fails to separate real actives from
property-matched decoys. This **downgrades the central positive** — "selection confers specificity" should
be restated as "selection enriches shape-fit binders; the *selectivity* interpretation is not supported."
The real signal in docking is **affinity, not selectivity** (proposal §4, row 3).

## Caveats (important — do not overclaim)

- **n=8 targets / ~25 mols each; AUROCs and the 8-target delta are noisy.** The actives-vs-decoys results
  are *ns*, not a proven reversal. Tier-1 falsifies (no clean separation); it cannot *confirm*.
- **"Known actives" are binders, not certified *selective* compounds** — some real drugs are genuinely
  promiscuous, which could depress their own-preference legitimately. This is exactly why **Tier-2**
  (correlate docked Δscore vs *measured* Δaffinity across target pairs) is the decisive calibration; it
  is not yet run.
- Orthosteric crystal pocket only; allosteric sites (where selectivity is often achievable) untested.
- Per-column z was computed over a small 8-target panel (noisier than the 20–41-target powered matrix).

## Bottom line

The instrument the whole project's specificity claims rest on **does not measure molecular selectivity** —
it measures shape-fit affinity, and the "specificity" contrast is largely a normalization artifact. This
does not erase the *generation-null* finding (that stands, and is even cleaner now: the generator makes
good binders, targeting was never the achievable axis), but it **reframes the one positive**: selection
gives you binders, not selective binders. Next: Tier-2 measured-selectivity calibration to confirm, and —
if confirmed — a learned selectivity oracle becomes the only credible path to real targeting.
