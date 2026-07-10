# SP-CS: Consensus-Scorer Discrimination Benchmark — Results

**Date:** 2026-07-11 · Branch `sp-cs-consensus` · Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-10-consensus-scorer-benchmark*.md`

## The question

SP-L/SP-F found that single docking/co-folding proxies disagree (smina-hacking; direction flips on 3/5
SP-F targets), motivating a **consensus scorer**. This benchmark tests the cleanest ground-truthed
version: does a consensus of **smina + Boltz** separate **known binders from random decoys** better — and
more **robustly** (worst-case AUROC) — than either scorer alone?

## Method

Per target, over its known∪random molecules: smina from `dock_scores.csv` (own-pocket) and Boltz
`affinity_pred` from `boltz_controls` (co-folded into each target's sequence; 78/78 cells, 0 failures).
Strength = `−score` (lower score = stronger). Per-target **AUROC** (known=positive) for smina, Boltz, and
two consensus rules — **rank-mean** (Borda, scale-free) and **z-sum** — with **mean** and **worst-case
(min)** across targets. 4 usable targets (`O43570_WT, P10721_WT, P02753_WT, P0C559_WT`); P06537 dropped
(only 3 knowns).

## Result — consensus does not beat Boltz

| target | smina | Boltz | rank-mean | z-sum | (known/random) |
|--------|-------|-------|-----------|-------|----------------|
| O43570_WT | 0.785 | 0.800 | 0.800 | 0.820 | 10 / 10 |
| P02753_WT | 0.905 | 1.000 | 0.990 | 1.000 | 10 / 10 |
| P0C559_WT | 0.944 | 1.000 | 1.000 | 1.000 | 8 / 10 |
| P10721_WT | 1.000 | 1.000 | 1.000 | 1.000 | 10 / 10 |
| **mean** | 0.908 | **0.950** | 0.948 | 0.955 | |
| **worst-case** | 0.785 | **0.800** | 0.800 | 0.820 | |

**Rank-mean consensus never exceeds Boltz on any target, and is marginally *below* Boltz on the mean
(0.948 vs 0.950).** Rank-averaging a weaker scorer (smina, mean 0.908) with a stronger one (Boltz, 0.950)
slightly *dilutes* the stronger — it does not help. z-sum's edge (mean +0.005, worst-case +0.020) is
within noise at N=4 with 8–10 molecules/target and is not a meaningful positive. **No worst-case
rescue:** consensus's worst-case (0.800) equals Boltz's; there was nothing to rescue.

## Why there was nothing to rescue

The worst-case-rescue hypothesis assumed a target where a single scorer *fails catastrophically*. That
did not occur here: **on real known drugs vs random decoys, smina never fails** (worst AUROC 0.785) —
both scorers discriminate real binders well. Smina's catastrophic failure mode is **smina-hacking on
smina-*optimized* candidates** (SP-F's P10721: smina −14 kcal/mol, Boltz-refuted, diversity-collapsed) —
a *candidate-optimization* phenomenon, not a known/random-discrimination one. So this benchmark's
molecule set structurally cannot reveal a consensus's hacking-guard value (see Future work).

## Verdict — prefer Boltz-validation over score-averaging

**Score-averaging consensus (smina + Boltz) provides no measurable benefit over Boltz alone, and
rank-mean marginally hurts.** The benchmark's supported positive is that it **independently re-confirms
Boltz as the competent scorer** (mean AUROC 0.950, ≥ smina on every target) — which *justifies the
Boltz-as-independent-validator pattern already used in SP-F/SP-L*. This sharpens the end-of-SP-F
recommendation: the evidence favors **using Boltz directly as the validator / selection scorer, not
averaging it with the weaker smina proxy.**

This closes the loop on the session's arc: generation-side levers don't confer targeting (SP2, SP-C,
SP-L, SP-F); selection carries the signal but is scorer-dependent; and the fix is **the better
independent scorer (Boltz) as validator**, not a cleverer generator and not score-averaging.

## Caveats

- **N = 4 targets**, 8–10 molecules/target; margins ≤ 0.02 AUROC are within noise. A powered claim needs
  more targets/molecules.
- Random "decoys" are REAL molecules that may include incidental weak binders — noise, applied equally to
  all scorers.
- The result is specific to **known-vs-random discrimination**; it does not test consensus in the
  **candidate-selection** regime where smina-hacking occurs (Future work).

## Future work (untested here, not a conclusion)

The consensus's hypothesized value — **guarding against smina-hacking during candidate selection** — is
untested by this benchmark but **directly checkable on data already in hand**: the SP-F P10721 treatment
candidates are smina-loved and Boltz-refuted, so a consensus (or Boltz-only) re-ranking of the SP-F
candidate pools would show whether it demotes the hackers smina promoted. That is a candidate-regime
benchmark, distinct from this known/random one.

## Reproduce

- Boltz known/random (proxy required for MSA; `BOLTZ=.venv-boltz/bin/boltz`):
  `env https_proxy=… .venv-boltz/bin/python -m scripts.boltz_controls --dock-scores data/dock/dock_scores.csv \
  --inputs data/boltz/matrix_inputs_sp_cs.json --scores data/dock/sp_cs_boltz_controls.csv --cap 10 --batch --no-kernels --accelerator gpu`
- Benchmark: `.venv/bin/python -m scripts.consensus_score --boltz data/dock/sp_cs_boltz_controls.csv \
  --targets O43570_WT,P10721_WT,P02753_WT,P0C559_WT`

Artifacts: `data/dock/sp_cs_boltz_controls.csv`, `scripts/consensus_score.py`.
