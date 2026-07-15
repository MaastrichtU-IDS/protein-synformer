# Contrastive Paralog-Discrimination Gate (Gate 1) — BORDERLINE POSITIVE (underpowered)

**Date:** 2026-07-15 · Branch `powered-specificity` · Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-15-contrastive-*` · Scripts: `scripts/contrastive_{data,train}.py`,
`scripts/discrim_eval.py`

> **Supersedes an earlier draft** that read a *collapsed* fine-tune (final-epoch train margin negative —
> an optimization failure, not a result) as "no transfer." Stabilized re-run below (full-batch gradient,
> length-normalized margin, best-train-margin checkpoint) is the real result.

## Question

The base model was trained positives-only, never asked to discriminate targets. Does a **contrastive**
objective — push route-LL higher under a target a drug *measurably binds* than a paralog it *measurably
does not* — produce **transferable** paralog discrimination on a **held-out kinase family (CSNK1) never
seen in training**?

## Method

- routed∩KIBA drugs, within-family measured binder(≥12.1)/non-binder(≤11.3) labels: **80 train triples
  (34 mols)** across MAPK/CDK/PRKC; **19 held-out CSNK1 triples (9 mols)**.
- Stabilized short fine-tune of SP-C: full-batch contrastive margin loss, **per-route length-normalized**
  LL, 40 epochs, lr 3e-5, **best-train-margin checkpoint saved** (not the final).
- Gate: held-out-CSNK1 paralog win-rate = fraction with LL(route|binder pocket) > LL(route|non-binder
  pocket); base vs FT; molecule-clustered bootstrap CI.

## Result — clean in-sample fit, and a suggestive (but underpowered) transfer signal

- **In-sample fit is now clean:** train-family margin(norm) rose monotonically **−0.53 → +3.83** (loss
  1.42 → 0.37). The model *can* learn contrastive paralog discrimination on the train families.
- **Held-out CSNK1 (unseen family):**

| | held-out CSNK1 win-rate |
|---|---|
| base SP-C | 0.263 |
| contrastive FT | **0.684**  (clustered CI **[0.500, 0.850]**) |
| Δ (FT − base) | **+0.421** |

**The point estimate (0.684) is well above chance and Δ is large — the first suggestive sign in the whole
project that a lever confers *transferable* paralog discrimination.** But it is **underpowered and not
conclusive**: the clustered CI lower bound sits exactly at 0.5 (n=19, 9 molecules), so the strict
pre-committed rule (CI must *exclude* 0.5) is not met.

## Verdict — promising, not proven; do NOT bank it, do NOT bin it

- **Not a clean PASS** (CI touches 0.5) → not yet a green light for full pretraining.
- **Not a null** → this must **not** go in the "does not work" column with the generator-side levers. It
  is qualitatively different: a large positive point estimate on a genuinely held-out family.
- **Firming required before any claim:** (1) **rotate the held-out family** (hold out MAPK / CDK instead
  of CSNK1) — does the transfer signal replicate, or was CSNK1 a fluke? (2) **multiple seeds**; (3) the
  base-CSNK1 win-rate (0.263) is *below* chance, which inflates Δ — read the absolute FT win-rate (0.684),
  not Δ, as the cleaner signal, and check whether base-below-chance is small-sample noise.

## Caveats

- **n=19 held-out triples (9 molecules)** — the binding power constraint; even a perfect run stays
  "suggestive" at this N. This is why the verdict is "firm it," not "conclude."
- One held-out family, one seed. KIBA is a coarse (integrated-score) label.
- The stabilized FT genuinely fits train (unlike the first collapsed run) — so the held-out lift is a real
  train→held-out generalization observation, not an artifact of a broken fit.

## Bottom line

Moving the discrimination signal *into the training objective* is the **first lever that shows a
suggestive transferable paralog-discrimination signal** (held-out CSNK1 win-rate 0.68 vs base 0.26) — but
at n=19 it is promising, not proven. The honest next step is the cheap **held-out-family rotation +
multi-seed** firming, not a capstone claim. Gate 2 (selective generation) remains downstream and untouched.
