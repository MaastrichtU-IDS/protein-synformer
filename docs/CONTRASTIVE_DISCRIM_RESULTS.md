# Contrastive Paralog-Discrimination Gate (Gate 1) — FAIL / inconclusive

**Date:** 2026-07-15 · Branch `powered-specificity` · Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-15-contrastive-*` · Scripts: `scripts/contrastive_{data,train}.py`,
`scripts/discrim_eval.py`

## Question

The base model was trained positives-only (route-reconstruction), so it was never asked to discriminate
targets. Does adding a **contrastive** objective — push route-LL higher under a target a drug *measurably
binds* than a paralog it *measurably does not* — produce **transferable** paralog discrimination, tested on
a **held-out kinase family (CSNK1) never seen in training**?

## Method

- Data: routed∩KIBA drugs with within-family measured binder/non-binder labels (KIBA ≥12.1 / ≤11.3).
  **80 train triples (34 mols)** across MAPK/CDK/PRKC; **19 held-out CSNK1 triples (9 mols)**.
- Short contrastive fine-tune of SP-C (margin loss, lr 1e-5, 3 epochs; reuse `dpo_train` +
  `ll_target_specificity` route/pocket LL + `get_log_likelihood_shortcut`).
- Gate: held-out-CSNK1 paralog **win-rate** = fraction of triples with LL(route|binder pocket) >
  LL(route|non-binder pocket), base vs fine-tuned, molecule-clustered bootstrap CI.

## Result — no transferable paralog discrimination

| | held-out CSNK1 win-rate |
|---|---|
| base SP-C | 0.421 |
| contrastive FT | **0.526**  (clustered CI **[0.33, 0.71]**) |
| Δ (FT − base) | +0.105 |

- **0.526 is essentially chance (0.5), and the CI [0.33, 0.71] spans chance widely** (n=19). No evidence of
  transferable paralog discrimination.
- **Training was unstable** — train-family margin rose (epoch 1 +1.79) then thrashed (epoch 2 −0.14; loss
  4.96→6.73→4.05), a noisy short fit on 80 triples — and **held-out never moved off chance**. This is the
  **SP-DPO pattern**: fits (noisily) in-sample, does not transfer.

## Verdict (pre-committed rule)

PASS required held-out win-rate above chance **and** above base with the clustered CI **excluding 0.5**.
The CI includes 0.5 → **FAIL / inconclusive → do NOT proceed to full contrastive pretraining.**

Even moving the discrimination signal *into the training objective* — the one lever the whole project had
not tried — does not produce transferable paralog discrimination at reachable scale. It joins the
generator-side nulls: the model can be nudged to fit a target-discrimination in-sample, but it does not
generalize to unseen paralogs.

## Caveats (do not over-claim FAIL either)

- **n=19 held-out triples (9 molecules)** — tiny; a null here is "no signal at this scale," **not proof of
  impossibility.** The binding constraint is measured within-CSNK1 paralog-discriminating drugs that also
  have routes (only 9).
- **The short FT was unstable** (lr/epochs/80-triples) — a larger, better-tuned run *might* fit train more
  cleanly, but (a) that starts to become the full pretraining this gate exists to justify, and (b)
  held-out N caps what any run can demonstrate here.
- Rotating the held-out family (MAPK/CDK) is the cheap robustness follow-up if this were to be revisited;
  the prior (SP-DPO no-transfer, DAVIS coin-flip, oracle domain-wall) makes a transfer signal unlikely.

## Bottom line

The training-objective lever — contrastive paralog discrimination — **does not clear the transfer gate**
with available data. Consistent with the project thesis: no reachable computational lever (conditioning,
enrichment, DPO, learned oracle, allosteric, or now a contrastive objective) confers transferable
paralog targeting; the wall is the absence of a *learnable, transferable* selectivity signal at the scale
and data available. Gate 2 (selective generation) is moot without Gate 1.
