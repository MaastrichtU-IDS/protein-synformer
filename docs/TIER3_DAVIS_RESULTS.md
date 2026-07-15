# Tier-3 (DAVIS): properly powered, kinase docking-selectivity is a coin flip at the pair level

**Date:** 2026-07-15 · Branch `powered-specificity` · Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-15-davis-kinase-calibration*.md` · Scripts:
`scripts/davis_{prep,analyze}.py`, `scripts/davis_dock_driver.sh` · Follows
[TIER2_CALIBRATION_RESULTS](TIER2_CALIBRATION_RESULTS.md)

## Question

Firm-or-break Tier-2's within-kinase docking-selectivity ρ 0.245 — which rested on **3 kinase pairs / 320
triples** from a noisy ChEMBL scrape — on the **DAVIS** kinome panel: dense, single-source (low-noise) Kd
for 68 drugs, giving **78 protein-kinase pairs / 5,304 triples**.

## Method

68 DAVIS drugs docked into the 15 overlapping kinase crystal pockets (1,020 docks; ATP-site autobox;
DAVIS drugs are ATP-competitive). Per-pocket z over the 68 drugs; for every kinase pair (A,B) and drug
measured on both, Spearman ρ(measured ΔpKd, −docked Δz) — same convention as Tier-2, + = docking tracks
selectivity. Compound-clustered bootstrap; per-pair ρ distribution as the guard against a few pairs
carrying the pooled number.

## Result — at the level that matters, it is a coin flip

| set | kinases | pairs | triples | pooled ρ | clustered 95% CI | **per-pair positive** | **median per-pair ρ** |
|---|---|---|---|---|---|---|---|
| **PRIMARY (protein kinases)** | 13 | 78 | 5,304 | +0.085 | [+0.024, +0.146] | **46/78 (59%)** | **+0.095** |
| ROBUSTNESS (all) | 15 | 105 | 7,140 | +0.075 | [+0.016, +0.132] | 64/105 (61%) | +0.074 |
| **PRIMARY, failed poses dropped** | 13 | 78 | 4,936 | +0.084 | — | **43/78 (55%)** | **+0.037** |

**Lead with the pair — it is the honest unit of evidence.** The question that matters is "can docked
selectivity rank *one* kinase's paralog selectivity over a sibling?" At that level it is **essentially a
coin flip**: 43–46 of 78 pairs positive (55–59%, binomial p≈0.1–0.25, **not** significant above chance),
median per-pair ρ **+0.037 to +0.095 ≈ 0**.

**The pooled ρ ≈ 0.085 is "significant" only because n is enormous.** 5,304 triples are heavily
non-independent (68 shared drugs, 78 pairs from 13 kinases sharing pockets), which inflates the pooled
power; the tight CI is an artifact of triple-count, not a usable effect. In absolute terms ρ ≈ 0.085 is
**<1% of selectivity rank-variance** (0.085² ≈ 0.7%). Robust to failed-pose removal (32/1020 clashing
score>0 poses dropped → pooled ρ 0.085→0.084, unchanged; pair-level if anything *weaker*).

## Verdict

**At the pair level — the level that matters for selecting one paralog over another — docking-selection is
essentially a coin flip (≈55–59% of pairs positive, median per-pair ρ ≈ 0).** The pooled ρ ≈ 0.085 is
statistically "significant" only because 5,304 non-independent triples inflate the power; it is <1% of
variance and not a usable signal. So the project's one positive, tested on properly-powered ground truth,
is **not a usable kinase-selectivity predictor** — it is a whisper detectable only in aggregate.

## What this changes

- CAPSTONE / FINDINGS: replace the ρ 0.245 / "~6% variance / real but weak" figure (a 3-pair thin-data
  artifact) with the honest statement: **at the pair level docking-selection is a coin flip (median
  per-pair ρ ≈ 0, 55–59% positive); the pooled ρ ≈ 0.085 is significant only by triple-count.** This
  *strengthens* the overall thesis — even the one apparent positive is, properly powered, practically
  null.
- Bringing in the dedicated panel was the right call: the project would otherwise have stood behind an
  inflated 0.245.

## Caveats

- Some docks return clashing/failed poses (positive smina scores seen in logs) → noise that biases ρ
  *downward*; the true signal may be marginally higher than 0.085, but not to 0.245.
- Docking into heterogeneous crystal structures (each kinase's own bound inhibitor defines the autobox);
  fixed per pocket, differences out in the z.
- DAVIS Kd is a single high-quality assay source (a strength — low label noise). Still a proxy;
  physics-based, rigid, orthosteric.
- Protein-kinase-family, ATP-site only — says nothing about GPCRs (Tier-2: null) or other classes.

## Reproduce

```
.venv/bin/python -m scripts.davis_prep
bash scripts/davis_dock_driver.sh          # ~1020 docks, ~2h
.venv/bin/python -m scripts.davis_analyze
```
Artifacts: `data/dock/davis/{dock_set.txt, measured_davis.json, kinase_pockets.json, dock_scores.csv,
davis_summary.json}`.
