# Contrastive Paralog-Discrimination Training (gated) — Design

**Date:** 2026-07-15 (rev. after KIBA paralog feasibility scout) · Sub-project: attack targeting at the
*training objective*. The base model was trained positives-only (route-reconstruction), so it was never
asked to discriminate targets — which is why no downstream lever conferred selectivity. Add a
**contrastive** objective (a route should be more likely under a target it *measurably binds* than a
paralog it *measurably does not*), gated by a cheap short-fine-tune **held-out-family paralog transfer
test** before any expensive full pretraining. · Depends on: SP-C pocket model + `get_log_likelihood`
(`dpo_train` infra), KIBA (routed∩KIBA), existing pocket `.npz` set, `ll_target_specificity` metric.

## 1. Goal & two-gate framing

> Can a contrastive objective make the pocket model's route-likelihood **transferably** discriminate
> **paralogs** — tested on a **held-out kinase family never seen in training**?

- **Gate 1 (this spec): transferable paralog discrimination.** Short contrastive fine-tune on TRAIN
  families → probe a HELD-OUT family's paralog discrimination. Full pretraining only if it clears the gate.
- **Gate 2 (deferred): selective generation.** Transferable discrimination is necessary, not sufficient
  (SP-DPO: route-LL preference ≠ selective samples). Out of scope until Gate 1 passes.

**Why gated:** contrastive training is SP-DPO's objective moved into pretraining; SP-DPO fit in-sample but
did not transfer. Gate on held-out transfer, cheaply, first.

## 2. Feasibility (resolved by scout)

- **Paralog discrimination exists in KIBA** (unlike CSNK1-in-DAVIS, which was 0): among **313 routed∩KIBA
  drugs**, within-family drugs that bind one paralog (KIBA ≥12.1) and not a sibling (≤11.3):
  **MAPK 15, CDK 7, CSNK1 7, PRKC (isoform-pair) up to 20, NEK 6, RPS6KA 5**.
- **Pockets already exist** (1,324-pocket set): **MAPK all 9 isoforms**, CDK 8/9, CSNK1 all 6, PRKC 6/8 —
  no new structure-building needed for MAPK/CSNK1 (the "panel expansion" is mostly already on disk).
- **Routes** for these drugs come from `filtered_pathways` (they are routed∩KIBA by construction).

## 3. Data & labels

- **Molecules:** routed∩KIBA drugs (313) with within-family measurements.
- **Families / pockets:** train families = **MAPK (9), CDK (8), PRKC (6)**; **held-out family = CSNK1 (6
  isoforms)** — chosen because it has discriminating drugs (7) and full pocket coverage, and is a *distinct*
  family so no CSNK1 sibling is ever seen in training (the honest paralog-transfer test). (Held-out family
  can be rotated as a robustness check.)
- **Binder / non-binder** per (drug, kinase) from KIBA score: binder ≥ 12.1, non-binder ≤ 11.3, drop the
  middle. Coarser than pKd (KIBA is an integrated score) — documented caveat.
- **Contrastive triples:** (route, binder-isoform pocket, non-binder-isoform pocket) **within a family**,
  built from TRAIN families only.

## 4. Objective (reuse `dpo_train` infra)

Margin loss on route-conditioned log-likelihood, varying the **target** axis with **measured** labels:

  `L = mean( softplus( margin − ( LL(route | pocket_binder) − LL(route | pocket_nonbinder) ) ) )`

`LL = get_log_likelihood(code(pocket), route)["total"].sum(dim=1)`. Policy = SP-C (trainable); short run
(few hundred steps, lr 1e-5, AdamW); log train margin + a held-out-family margin monitor; save via
`build_out_checkpoint`.

## 5. Transfer gate (the decision) — adapt `ll_target_specificity.py`

After the short fine-tune, on the **HELD-OUT family (CSNK1)**: pairwise win-rate = fraction of
within-CSNK1 discriminating (drug, binder-isoform, non-binder-isoform) triples where
`LL(route|binder) > LL(route|non-binder)`, vs chance (0.5), vs the base SP-C model, with a
**molecule-clustered bootstrap** CI. This is *pure paralog transfer* — no CSNK1 isoform was in training.

## 6. Decision criteria (pre-committed)

- **PASS → full contrastive pretraining:** held-out-family (CSNK1) paralog win-rate rises **meaningfully
  above chance and above base SP-C**, CI (clustered) excluding 0.5.
- **FAIL → stop:** train-family win-rate rises but held-out CSNK1 stays at chance (SP-DPO pattern) →
  paralog discrimination doesn't transfer to an unseen family; contrastive pretraining won't confer it.
- **Honesty guards:** report train-family vs held-out-family side by side (the gap is the finding);
  held-out CSNK1 has 7 discriminating drugs → small N, so a null is "no signal at this scale," not proof;
  rotate the held-out family (CDK/MAPK) as a robustness check if Gate 1 is borderline.

## 7. Components & interfaces

| file | responsibility |
|---|---|
| `scripts/contrastive_data.py` | routed∩KIBA drugs; canonical-SMILES route lookup; UniProt→gene→family map; binder/non-binder from KIBA score; TRAIN-family / HELD-OUT-family split; within-family contrastive triples. Pure `binder_label`, `make_within_family_triples` TDD'd |
| `scripts/contrastive_train.py` | short contrastive fine-tune (reuse `dpo_train` model-load + `get_log_likelihood`; `contrastive_loss` TDD'd on toy LLs) → fine-tuned ckpt |
| `scripts/discrim_eval.py` | held-out-family paralog win-rate (adapt `ll_target_specificity`), base-vs-FT, molecule-clustered bootstrap |
| `docs/CONTRASTIVE_DISCRIM_RESULTS.md` | Gate-1 verdict |

## 8. Testing (TDD)

- `binder_label(kiba)`: ≥12.1 binder, ≤11.3 non-binder, else None.
- `contrastive_loss(ll_bind, ll_nonbind, margin)`: decreases as (ll_bind − ll_nonbind) exceeds margin;
  positive/finite (toy tensors).
- `make_within_family_triples`: only within-family pairs; only TRAIN families in training triples;
  held-out family excluded; a drug binding one isoform + not a sibling yields a triple.
- route lookup: canonical-SMILES match into `filtered_pathways`; missing → skipped.

## 9. Caveats / non-goals

- **KIBA score is a coarse integrated metric** (not pKd), narrow range; binder/non-binder threshold
  (12.1/11.3) is the standard binarization but noisier than Kd.
- **Small held-out N** (CSNK1: 7 discriminating drugs) — Gate 1 is a cheap directional test; a null is
  "no signal at this scale," not proof; rotate held-out family for robustness.
- **Transfer is the whole risk** (SP-DPO precedent) — exactly what Gate 1 measures.
- **Discrimination ≠ selective generation** (Gate 2, deferred).
- **No full pretraining in this spec** (gated). Pockets for held-out/train families already exist; the
  2 missing PRKC/CDK isoforms are simply excluded (not built).
- Reuse `dpo_train` / `ll_target_specificity`; no new training/eval scaffolding.
