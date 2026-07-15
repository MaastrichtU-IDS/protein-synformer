# Tier-3 (DAVIS): the kinase docking-selectivity signal is real but ~3× weaker than Tier-2 implied

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

## Result — keep the sign, break the magnitude

| set | kinases | pairs | triples | pooled ρ | clustered 95% CI | per-pair positive | median per-pair ρ |
|---|---|---|---|---|---|---|---|
| **PRIMARY (protein kinases)** | 13 | 78 | 5,304 | **+0.085** | **[+0.024, +0.146]** ✓ excl. 0 | 46/78 (59%) | +0.095 |
| ROBUSTNESS (all) | 15 | 105 | 7,140 | +0.075 | [+0.016, +0.132] ✓ | 64/105 (61%) | +0.074 |

- **The signal is REAL** — the pooled ρ CI excludes 0 on 5,304 triples, and it is *diffuse* (median
  per-pair ρ +0.095 ≈ pooled, so no single pair carries it — unlike the Tier-2 concentration risk).
- **But it is far weaker than Tier-2 implied:** ρ ≈ **0.085**, not 0.245 — a ~3× reduction. That is
  **<1% of selectivity rank-variance** (0.085² ≈ 0.7%). At the individual-pair level only **59% of pairs
  are positive** — not significantly above a coin flip (binomial 46/78, p≈0.1). Tier-2's 0.245 was
  thin-data optimism (3 pairs, one noisy panel).

## Verdict

**FIRM that it is real; BREAK the magnitude.** Properly powered on dense, low-noise ground truth, docked
own-vs-mismatch selectivity tracks measured kinase-paralog selectivity to a **statistically detectable but
practically negligible** degree (ρ ≈ 0.085). The project's one positive stands as *real* but must be
reported at its true, much smaller effect size — not the 0.245 the capstone provisionally carried.

## What this changes

- CAPSTONE / FINDINGS: the kinase docking-selection signal is **ρ ≈ 0.085 (~0.7% variance), 59% of pairs
  positive** — "real but practically negligible," replacing the ρ 0.245 / "~6% variance" figure, which was
  a 3-pair artifact. The overall thesis (targeting is very hard; only a weak physics signal is reachable)
  is *strengthened*, and the one positive is now correctly sized.
- Bringing in the dedicated panel was the right call: it materially corrected the headline number the
  project would otherwise have stood behind.

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
