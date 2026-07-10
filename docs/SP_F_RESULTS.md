# SP-F: Fragment-Seeding Hill-Climb — Results

**Date:** 2026-07-10 · Branch `sp-f-fragment-seeding` · Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-10-fragment-seeding-hillclimb*.md`

## The question

SP-L showed that *motif-enrichment* of the frozen generator toward docking-winners is a structural
no-op (winners use the generator's modal motifs — nothing distinctive to amplify). SP-F tried the one
generation-side lever that does **not** depend on winners being distinctive — **local search around a
specific good binder**:

> Dock a pool → take the top-k binders → generate their **synthesizable neighbors** with SynFormer's
> analog sampler (conditioned on each seed *molecule*) → dock the neighbors → re-seed on the new top-k →
> iterate. Does exploring a good binder's synthesizable neighborhood produce better binders, over a
> docking-budget-matched control?

The pocket enters only via docking-selection of the seed; generation is seed-molecule-conditioned
(base `sf_ed_default.ckpt`, fetched from HF). Three budget-matched arms decompose the effect:
**treatment** (analog-seed on the best dockers), **control-A** (analog-seed on *random* dockers —
isolates docking guidance), **control-B** (fresh SP-C pocket draws — isolates the analog mechanism).

## Method & scale

Full implementation (analog `featurize_stack` restore, `generate_analogs.py`, 3-arm resumable
`fragment_loop.py`), unit-tested, per-task reviewed, whole-loop dry-run-validated on the box. Shakedown:
`O43570_WT`, `P06537_WT` at 3 arms; extended to `P10721_WT`, `P02753_WT`, `P0C559_WT` at 2 arms
(treatment + control-B) to power the corroboration check. All arms `R=2`, `B=60` docks/round, `k=3`
seeds, shared round-0 docked once per target. Final top-10 per arm re-scored with **Boltz-2**
(independent co-folding, `api.colabfold.com` MSA).

## Docking (smina) result — treatment beats fresh-draw control on 4/5

Final-round top-10 mean docking score, treatment vs control-B (Δ = treatment − control-B; − = better):

| target | treatment | control-B | Δ (smina) | hill-climb r0→r1 |
|--------|-----------|-----------|-----------|-------------------|
| O43570_WT | −8.47 | −8.14 | −0.33 | flat |
| P06537_WT | −8.93 | −8.49 | −0.44 | flat (r1<r0) |
| P10721_WT | −13.79 | −11.45 | **−2.34** | improves (−12.5→−13.8) |
| P02753_WT | −9.76 | −10.68 | **+0.92** | worse |
| P0C559_WT | −9.51 | −8.76 | −0.75 | improves |

By smina, treatment wins 4/5 (mean Δ ≈ −0.59). The **3-arm decomposition** (O43570 / P06537) is clean and
consistent: treatment − control-A = −0.87 / −1.85 (docking-guided seeding ≫ random-seed), control-A −
control-B = **+0.54 / +1.41** (analog-searching a *random* seed is *worse* than fresh draws — you get
trapped near a mediocre molecule). So **the value is the docking guidance, not the analog mechanism**;
the mechanism alone hurts.

## Boltz-2 corroboration — the smina edge does NOT survive (net-null)

Final top-10 mean Boltz `affinity_pred` (lower = stronger), treatment vs control-B:

| target | Δ (Boltz) | Boltz verdict | agrees with smina? |
|--------|-----------|---------------|--------------------|
| O43570_WT | −0.38 | treatment better | ✅ agree |
| P0C559_WT | −0.40 | treatment better | ✅ agree |
| P02753_WT | −0.35 | treatment better | ❌ (smina said worse) |
| P06537_WT | +0.96 | treatment worse  | ❌ (smina said better) |
| P10721_WT | +0.45 | treatment worse  | ❌ (smina said much better) |

By Boltz, treatment is favored on **3/5** targets — by count, by **median Δ (−0.349)**, and by
best-binder-found (treatment's strongest beats control-B's on the same 3 targets). But the effect is
**not consistent**: it flips on 2/5, and the **mean Δ (+0.055) is ≈ 0 only because P06537's +0.955
outlier cancels the rest** — so no single aggregate should carry the verdict. The robust reading is
**scorer disagreement, not scorer refutation**: smina and Boltz flip direction on 3/5 targets, and
smina's *magnitude* fails to predict Boltz's. Most tellingly, the **largest smina win — P10721 (−2.34)
— is exactly where Boltz most disagrees (+0.45)**: control-B's fresh draws held Boltz's strongest binder
(aff −1.459), which the smina-guided hill-climb steered *away* from while collapsing to a narrow,
high-smina neighborhood (scaffold diversity 0.30, scores ≈ −14 kcal/mol) — smina over-scoring a greasy
local basin, a docking artifact rather than a real binder.

## Verdict

**The fragment-seeding loop mechanically works and concentrates *smina*-good molecules — docking guidance
demonstrably helps (treatment ≫ random-seed).** Under co-folding the loop is *weakly* favorable
(Boltz prefers treatment on 3/5 by count, median, and best-binder) but **inconsistent** (flips on 2/5),
and — the operative failure — **smina's magnitude does not predict Boltz's**: the loop's biggest
smina win is its worst co-folding disagreement. So the smina-measured advantage is **not a reliable
indicator of real (co-folding) benefit**. The P10721 case shows *how* this fails — hill-climbing smina
walks into a diversity-collapsed, co-folding-refuted greasy basin (the project's finding #1 —
"optimizing a rigid proxy invites proxy-hacking" — manifesting on the flagship win, though 3/5 targets
came out fine by both scorers). A full powered run was **not** justified: chasing a 3/2-split,
scorer-disagreeing, artifact-tainted effect over multiple days would amplify a smina-specific signal the
independent scorer already flags as unreliable.

This joins the project's throughline. Generation-side levers — conditioning (SP2 sequence, SP-C pocket),
motif re-biasing (SP-L), and now local search (SP-F) — do **not** robustly confer better/target-specific
binding. What consistently carries signal is **selection against the 3D pocket**; and even that is
scorer-dependent (smina and Boltz disagree), so single-proxy selection is not enough — the honest path
forward is **consensus / independent-scorer validation**, not a cleverer generator loop.

## Caveats

- N = 5 targets, `R = 2`, `B = 60` (shakedown scale); a properly-powered study would tighten CIs, but the
  co-folding **direction is inconsistent (3/2 split, scorers disagree on 3/5)** and the mechanism-level
  reads (guidance helps smina; analog-on-random hurts; the P10721 diversity-collapse artifact) generalize.
- The analog sampler optimizes *similarity to the seed*, not docking; the loop injects docking only via
  re-seeding. A reward-tuned analog objective is untried.
- smina is a rigid shape-fit proxy; the whole point here is that its wins are **not** corroborated by
  Boltz — which is the result, not a limitation of the comparison.
- control-A (random-seed) was run only on 2 of 5 targets; the 3-arm decomposition is N=2.

## Reproduce

- Loop (box, `SMINA=$(pwd)/smina.static`): `python -m scripts.fragment_loop --pocket-ckpt <SP-C> \
  --targets <json> --arms treatment,control_a,control_b --rounds 2 --budget 60 --k 3 --out-dir data/dock/sp_f`
- Decomposition: `python -m scripts.sp_f_analyze --summary data/dock/sp_f/loop_summary.csv`
- Boltz corroboration (**proxy required for MSA**): `env BOLTZ=.venv-boltz/bin/boltz https_proxy=… \
  .venv-boltz/bin/python -m scripts.sp_f_boltz --targets <…> --arms treatment,control_b --m 10`

Artifacts: `data/dock/sp_f/loop_summary.csv`, `data/dock/sp_f_boltz_scores.csv`, per-arm/round trees
under `data/dock/sp_f/`. Base model `data/trained_weights/sf_ed_default.ckpt` (HF `whgao/synformer`).
