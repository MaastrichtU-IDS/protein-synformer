# Contrastive Target-Discrimination Training (gated) — Design

**Date:** 2026-07-15 · Sub-project: attack targeting at the *training objective* — the base model was
never trained to discriminate targets (positives-only route-reconstruction), so no downstream lever could
extract a signal the loss never created. Add a **contrastive** objective (a route should be more likely
under its *true* target's pocket than a *measured non-binder's*), but make the expensive full pretraining
**earn its way in** through a cheap short-fine-tune **transfer gate**. · Depends on: SP-C pocket model +
`get_log_likelihood` (via `dpo_train` infra), DAVIS + ChEMBL measured panels, the precomputed synthesis
pathways.

## 1. Goal & two-gate framing

> Can a contrastive objective make the pocket-conditioned model's route-likelihood **transferably**
> discriminate a known binder's true target from measured non-binders on **held-out** kinases (ideally
> paralogs)?

- **Gate 1 (this spec): transferable discrimination.** Short contrastive fine-tune on TRAIN kinases →
  re-probe HELD-OUT kinases (paralog + cross-family). Only if held-out discrimination rises meaningfully
  above chance do we commit to full contrastive pretraining.
- **Gate 2 (deferred): selective generation.** Transferable discrimination is necessary but not
  sufficient — SP-DPO showed route-LL preference need not become selective *samples*. Out of scope until
  Gate 1 passes.

**Why this is the right lever and why it's gated:** the discrimination probe was near-uninformative by
design (a model never trained to discriminate barely discriminates: 54%, ns) — it confirms the diagnosis,
not the ceiling. The real risk is **transfer**: contrastive training is SP-DPO's objective moved into
pretraining, and SP-DPO fit in-sample (margin 2.97→3.49) but **did not transfer** to held-out pockets. So
we gate on held-out transfer, cheaply, before the expensive run.

## 2. Feasibility (resolved)

- **No molecule→route base model exists** (only pocket + sequence checkpoints), so we **cannot project
  arbitrary SMILES to routes.** Instead use molecules that **already have routes** in
  `filtered_pathways_370000.pth` (67,429 canonical) **and** measured multi-target binding:
  - **routed ∩ DAVIS = 22 drugs** — each with a route AND a *dense* 15-kinase Kd profile (clean measured
    binders and non-binders). This is the clean core.
  - **routed ∩ Tier-2 multi-target = 90**; **routed ∩ any-ChEMBL-panel-measured = 1,337** — broader but
    sparser measurement.
- **Clean measured negatives** (the advisor's correction — not Papyrus-absence, which is *untested*):
  DAVIS-dense (non-binding is measured, pKd ≈ 5) is the primary negative source; ChEMBL pairs where the
  molecule is measured on ≥2 panel targets supplement it.

## 3. Data & pairs

- **Molecules:** routed ∩ measured (start with the DAVIS-dense 22 + the 90 Tier-2; expand to the routed
  ∩ ChEMBL ≥2-measured set for more if needed). Routes from `filtered_pathways` (keyed by canonical SMILES).
- **Targets:** the 13–15 panel kinases (pockets already prepped). **Split kinases** into TRAIN (~10) and
  HELD-OUT (~5, including paralog pairs, e.g. hold out one of KIT/JAK3/CDK5's siblings) — the split is over
  *targets*, not molecules, so held-out discrimination tests target generalization.
- **Binder / non-binder** per (molecule, kinase): binder = pKd ≥ 7 (Kd ≤ 100 nM); non-binder = pKd ≤ 5
  (DAVIS non-binder floor / Kd ≥ 10 µM). Drop the ambiguous middle. Pure, TDD'd.
- **Contrastive triples:** (route, binder-kinase pocket, non-binder-kinase pocket) built from TRAIN
  kinases only for training; HELD-OUT kinases reserved for the transfer probe.

## 4. Objective (reuse `dpo_train` infra)

Margin/contrastive loss on route-conditioned log-likelihood (the only change vs `dpo_train` is which
axis varies — **target**, with measured labels, not molecule):

  `L = mean( softplus( margin − ( LL(route | pocket_binder) − LL(route | pocket_nonbinder) ) ) )`

i.e. push the route's likelihood higher under a pocket it *measurably binds* than one it *measurably does
not*, for the same route. `LL = get_log_likelihood(code(pocket), route)["total"].sum(dim=1)`. Policy =
SP-C (trainable); short run (few hundred steps), small lr (1e-5), AdamW, log the train margin and a
held-out-margin monitor. Reuse `build_out_checkpoint` for a `load_model`-compatible save.

## 5. Transfer gate (the decision) — reuse/adapt `ll_target_specificity.py`

`scripts/ll_target_specificity.py` already computes exactly the discrimination metric (LL of a molecule's
pathway under its true target vs decoys: top-1 accuracy, mean rank, pairwise win-rate) for the sequence
models — **adapt it to the pocket model and to measured binder/non-binder targets.** After the short
fine-tune, on **HELD-OUT kinases**, compute pairwise win-rate = fraction where LL(route | measured-binder)
> LL(route | measured-non-binder), **stratified paralog vs cross-family**, vs the pre-FT baseline and vs
chance (0.5). Report base SP-C vs contrastively-fine-tuned, held-out.

## 6. Decision criteria (pre-committed)

- **PASS → full contrastive pretraining:** held-out pairwise win-rate rises **meaningfully above chance
  and above the base model** — ideally on **paralog** pairs, with a bootstrap CI (clustered by molecule)
  excluding chance.
- **FAIL → stop (SP-DPO pattern):** train-kinase win-rate rises but **held-out stays at chance** →
  discrimination doesn't transfer; contrastive pretraining would not confer targeting. Report and stop.
- **Honesty guards:** report train vs held-out side by side (the gap *is* the finding); small-N held-out
  is inconclusive-not-impossible; paralog and cross-family reported separately (cross-family lift with
  paralog-null is a weaker pass than paralog lift).

## 7. Components & interfaces

| file | responsibility |
|---|---|
| `scripts/contrastive_data.py` | routed ∩ measured molecules; canonical-SMILES route lookup; binder/non-binder labels from DAVIS/ChEMBL pKd; TRAIN/HELD-OUT kinase split; emit contrastive triples. Pure `binder_label(pkd)`, `make_triples(...)` TDD'd |
| `scripts/contrastive_train.py` | short contrastive fine-tune (reuse `dpo_train` model-load + `get_log_likelihood`; new `contrastive_loss` TDD'd on toy LLs) → fine-tuned ckpt |
| `scripts/discrim_eval.py` | adapt `ll_target_specificity` to pocket + measured binder/non-binder; held-out win-rate, paralog/cross-family, base-vs-FT, clustered bootstrap |
| `docs/CONTRASTIVE_DISCRIM_RESULTS.md` | Gate-1 verdict |

## 8. Testing (TDD)

- `binder_label(pkd)`: ≥7 → binder, ≤5 → non-binder, else None.
- `contrastive_loss(ll_bind, ll_nonbind, margin)`: decreases as (ll_bind − ll_nonbind) grows past margin;
  positive/finite; toy tensors.
- `make_triples`: only TRAIN kinases in training triples; a molecule with a binder+non-binder yields a
  triple; held-out kinases excluded from training.
- route lookup: canonical-SMILES match into `filtered_pathways`; missing → skipped.

## 9. Caveats / non-goals

- **Small N** (22 DAVIS-dense + ~90 Tier-2 routed; more from ChEMBL but sparser) — Gate 1 is a *cheap
  directional test*, not a definitive negative; a held-out null is "no signal at this scale," not proof.
- **Transfer is the whole risk** (SP-DPO precedent) — that's exactly what Gate 1 measures.
- **Discrimination ≠ selective generation** (Gate 2, deferred).
- **Kinase-only** (where routes∩measured exists and docking had any signal); GPCRs out of scope.
- **No full pretraining in this spec** (gated). **No molecule→route projection** (base model absent; we
  use routed∩measured molecules instead).
- Reuse `dpo_train`/`ll_target_specificity` rather than new training/eval scaffolding.
