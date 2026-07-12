# Tier-1 Calibration Results: the specificity metric has a real (modest) cross-family selectivity signal

**Date:** 2026-07-13 · Branch `powered-specificity` · Proposal:
`docs/SPECIFICITY_CALIBRATION_PROPOSAL.md` · Scripts: `scripts/tier1_{prep,analyze}.py`,
`tier1_dock_driver.sh`

> **Correction note:** an initial version of this doc led with a confounded number (a top-M + family-
> clustered-panel artifact) and wrongly concluded the specificity metric was falsified. The family-
> stratified recompute below (advisor-caught) reverses that. This is the corrected record.

## Question

Does the own-vs-mismatch docking **z-delta** — the metric behind the project's central positive
("selection against the pocket confers specificity", −0.6/−0.8) — track **real** selectivity? Tier-1 test:
dock **known actives**, **property-matched decoys** (MW/logP/HBD-matched, Tanimoto<0.35), and the model's
**generated candidates** (25 each) for 8 targets into a shared 8-pocket panel, and compare the metric
across classes.

## Clean finding 1 — smina has a modest, family-dependent own-pocket AFFINITY signal

Raw **own-pocket** score ranks known actives above property-matched decoys — **mean AUROC 0.66**, strongly
pocket-dependent: KIT 0.80, CDK5 0.74, JAK3 0.73, 5-HT1A 0.73, 5-HT2A 0.65, A1R 0.58, **RAB9A 0.56, CA12
0.48** (smina fails on the shallow/polar CA zinc pocket and the GTPase — no metal/induced-fit handling).
Generated candidates dock **comparably to real actives** on the own pocket (KIT −9.0 vs −9.9; A1R −8.9 vs
−8.4) — the generator makes genuinely good shape-fit binders.

## The confound (why the naive whole-panel comparison misleads)

The 8-target panel is **family-clustered** (3 kinases + 3 aminergic GPCRs + 2 outliers). Real actives are
**legitimately cross-reactive within family** — a KIT inhibitor docks into JAK3/CDK5 — so 2 of each
active's 7 mismatch pockets are same-family targets it genuinely binds, which flattens its
own-preference. Property-matched decoys have no family to cross-react with, so under top-M-by-own-score
selection they look *artificially* specific. Averaging all mismatch pockets together (and adding top-M
selection, and the CA12/RAB9A docking-failure pockets) produced a spurious "actives look *less* specific
than decoys" reversal. The tell: the no-selection per-molecule metric already showed the correct direction
(actives −0.043 vs decoys +0.056); only top-M flipped it.

## Clean finding 2 — de-confounded (family-stratified, 6 working targets, per-molecule)

Dropping the two docking-failure pockets (CA12, RAB9A) and splitting the mismatch panel into same-family
vs cross-family (more negative = prefers own):

| contrast | actives | decoys | candidates | diff(act−dec) 95% CI | AUROC(act>dec) |
|---|---|---|---|---|---|
| **own vs cross-family** | **−0.107** | +0.146 | −0.072 | **−0.253 [−0.44, −0.08]** ✓ excl. 0 | 0.575 |
| own vs same-family | +0.003 | +0.190 | −0.062 | −0.186 [−0.39, +0.01] ns | 0.542 |

- **Cross-family: real actives prefer their own pocket significantly more than matched decoys** (CI
  excludes 0). The specificity metric carries a **real, modest cross-family selectivity signal** — it is
  not a pure normalization artifact.
- **Same-family: actives show no own-preference over sibling targets** (ns) — they genuinely cross-react
  at the paralog level, and the metric correctly reports this rather than inventing selectivity.

## Verdict (corrected; per pre-committed decision rule)

Tier-1 does **not** falsify the metric. De-confounded, the own-vs-mismatch z-delta **does** track real
selectivity at the **cross-family** level (modestly: AUROC ~0.58, diff CI excludes 0) and honestly reports
**within-family cross-reactivity**. So the project's central positive is **refined, not downgraded**:

> Docking-selection specificity is a **real but modest cross-family** signal (own pocket vs unrelated
> families), riding on smina's family-dependent affinity signal. It does **not** demonstrate
> **paralog-level** selectivity — the hard, valuable case — for real known binders here.

This is consistent with Boltz's molecule-level non-corroboration being about *magnitude/paralog* resolution
rather than the axis being pure noise.

## Caveats (do not overclaim in the other direction now)

- **Effect sizes are modest** (cross-family AUROC 0.575; the win is "CI excludes 0", not a strong
  separation). n=6 targets, ~25 mols/class.
- **"Known actives" are binders, not certified *selective* compounds** — so Tier-1 shows the metric
  isn't noise, but cannot quantify selectivity fidelity. **Tier-2 (correlate docked Δscore vs *measured*
  Δaffinity across target pairs) remains the decisive test** and is well-motivated by this result.
- **Paralog-level selectivity — the actually-valuable problem — is NOT demonstrated** (own-vs-same-family
  ns for actives). Whether the powered study's within-family candidate "specificity" (−0.77) is real or a
  selection artifact is now the sharp open question; Tier-2 within-family pairs would settle it.
- Orthosteric crystal pocket only; allosteric sites (where paralog selectivity is often achievable)
  untested. CA/GTPase need better physics than rigid smina.

## Bottom line

The instrument is **not** falsified — corrected for a family-clustering confound, it shows a real, modest
**cross-family** selectivity signal plus a family-dependent **affinity** signal. What it does **not** yet
support is **paralog-level** selectivity for real drugs, which is the valuable target. The **generation-
null stands and is cleaner** (the generator makes good binders; targeting was never the achievable axis).
Next: **Tier-2** measured-Δaffinity calibration, with explicit within-family (paralog) pairs — that is what
tells us whether real selectivity is reachable at all, and whether the within-family candidate signal was
real.
