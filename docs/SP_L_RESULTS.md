# SP-L: Enrichment-Selection Loop — Results

**Date:** 2026-07-10 · Branch `sp-l-enrichment-loop` · Spec:
`docs/superpowers/specs/2026-07-10-closed-loop-enrichment-selection-design.md`

## The question

The project established **two nulls** on target-specificity: conditioning the SynFormer generator on
target information — sequence (SP2) or 3D pocket (SP-C) — does not confer targeting; only **selecting**
docked candidates against the 3D pocket produced a modest, method-dependent signal. This sub-project
asked the sharper follow-up:

> On a **frozen** SP-C model, can iteratively **enriching** generation toward each round's
> docking-winners (building-block / reaction-template reweighting, no weight updates) concentrate
> better-binding — and eventually more target-specific — molecules, beyond what a docking-budget-matched
> uniform control gets from the same number of draws?

## Method

A closed loop over the frozen SP-C pocket model: generate a pool → drug-like/SA-gate → dock a fixed
budget with smina → take the top-k winners → mine their synthetic routes (`Stack.get_mol_idx_seq` →
building-block fpindex ids; `get_rxn_idx_seq` → reaction-template ids) for over-represented motifs →
add a promote-only `log(w)` bias (clip `w_max`) to the next round's reactant-selection and
reaction-template logits. Two **docking-budget-matched** arms, identical in every way except that the
`enrich` arm feeds the mined weights forward while the `uniform` control passes no weights — so any
difference is attributable to enrichment, not to drawing/docking more molecules. (Design details and the
reweight-hook mechanics: see the spec.)

The implementation (Tasks 1–7) is complete, unit-tested (24 SP-L tests; full suite 126 pass/1 skip),
per-task reviewed, whole-branch reviewed, and validated end-to-end on the box (real GPU generation via
`.venv-train` → real smina docking → correct promote-only weights). `enrich_weights=None` reproduces
baseline generation exactly, so the machinery is a safe no-op for every existing caller.

## Shakedown run (the decisive experiment)

Per the "smaller pilot first" decision, a 2-target shakedown was run before committing to the full
5-target/~19 h pilot: targets `O43570_WT` (CA12) and `P06537_WT` (GR), `R=3` rounds, `B=100` docks/
round/arm, `N=1000` pool/round, `k=25` winners.

**Per-round top-10 mean docking score (kcal/mol; more negative = stronger; own pocket):**

| target | arm | round 0 | round 1 | round 2 |
|--------|-----|---------|---------|---------|
| O43570_WT | enrich  | −7.84 | −8.17 | −7.99 |
| O43570_WT | uniform | −7.84 | −8.17 | −8.03 |
| P06537_WT | enrich  | −8.75 | −8.73 | −8.52 |
| P06537_WT | uniform | −8.75 | −8.76 | −8.52 |

- **Round 0 is identical across arms** — the shared seed and no-weights round-0 confirm the two arms
  start from the same baseline (a built-in control that the loop is symmetric).
- **From round 1 on, enrich and uniform are within noise** (≤ ~0.04 kcal/mol) at every round, on both
  targets.
- **No round-over-round gain:** round 2 is *worse* than round 1 in 3 of 4 arms. Each round is a fresh
  independent draw of 1000 candidates; the loop does not concentrate better dockers over rounds.

**Verdict at the loop level: NULL.** Enrichment produced no measurable improvement over the
budget-matched uniform control, and no monotonic gain across rounds. Because the arms do not differ on
the primary own-pocket score, the downstream all-pairs **specificity** readout is moot — there is no
differential targeting to measure — and was not run. Boltz-2 out-of-loop validation was likewise
unnecessary: there is no docking win to independently corroborate.

## Why — the mechanism diagnostic (docking-free)

Generation-only checks (no docking) explain the null. They rule out a plumbing bug (the arms *do*
differ, so weights are applied) and locate the cause in **efficacy**, at two levels.

**1. At the loop's actual settings (`w_max = 5`, `temperature_reactant = 0.1`) enrichment barely bites.**
Using round 0's mined weights to generate round 1, per-template frequencies were identical to three
decimals between the enrich and uniform pools, and the building-block channel moved trivially
(molecules touching a promoted BB: 10.2% enrich vs 8.4% uniform). At these settings the treatment is
effectively not administered — `temperature_reactant = 0.1` makes reactant selection near-argmax, so a
`log(w ≤ 5)` bias barely competes with the retrieval scores.

**2. Even at 30× weights the composition shift is small and lands on *already-common* motifs, not on the
rare distinctive winner motifs.** Regenerating O43570_WT with all weights scaled to 30× (n = 300 each,
seed 7), per-promoted-template frequency:

| template | round-0 weight | enrich (30×) | uniform | Δ |
|---------:|---------------:|-------------:|--------:|--------:|
| 4   | 3.64 | 0.010 | 0.010 | +0.000 |
| 107 | 1.90 | 0.007 | 0.007 | +0.000 |
| 85  | 1.90 | 0.020 | 0.020 | +0.000 |
| 31  | 1.97 | 0.053 | 0.053 | +0.000 |
| 9   | 1.69 | 0.107 | 0.090 | **+0.017** |
| 50  | 1.59 | 0.067 | 0.057 | +0.010 |
| 8   | 1.19 | 0.220 | 0.223 | −0.003 |

The **rare, highly-weighted** templates (4, 107, 85 — the ones a genuinely distinctive winner motif
would live in) do **not** rise despite the largest weights: a global `log(w)` logit bias cannot
manufacture a template the model rarely produces *in-context* (template applicability depends on the
current stack state). The only movement is a ~1–2pp bump on templates the model *already* favors
(tpl 9, 50). The building-block channel is likewise weak-but-not-dead: most promoted BBs never appear,
but a few do respond to strong weights (bb 32531: 0.007 → 0.040 at 30×).

**3. Top-M docking is at a quality ceiling.** Best-per-round score is flat across rounds and arms
(O43570 −8.8/−8.6/−8.9; P06537 −9.2/−9.3/−8.7) and top-10 means hover near −8 regardless of arm. The
frozen model's top-M already saturates what its chemistry can reach against these pockets, so the modest
composition shifts enrichment *can* produce have nowhere to push.

## Conclusion — a third null, consistent with the thesis

At its designed settings the frozen-model enrichment loop **produces no measurable improvement** in
own-pocket docking over a budget-matched uniform control, and no round-over-round gain. The diagnostic
shows why this is not merely a matter of turning up a knob: stronger weights shift composition only
modestly and mostly among motifs the model already favors, the rare distinctive motifs don't respond to
a global logit bias, and top-M is at a quality ceiling regardless. So there is no reason to expect the
full 5-target pilot to differ — hence it was not run.

This is consistent with the project's throughline: neither **conditioning** the generator (sequence,
SP2; 3D pocket, SP-C) nor **re-biasing** it toward its own docking-winners (SP-L) has moved
target-specificity; the only lever that ever has is **selection** against the 3D pocket.

## Caveats — scope of the claim

- **This is "no measurable benefit at tested settings + a top-M ceiling," not "structurally impossible."**
  Enrichment *can* shift composition when pushed hard (see bb 32531, tpl 9); what it cannot do here is
  (a) raise the rare distinctive motifs via a global logit bias, or (b) push top-M past the frozen
  model's quality ceiling. We did **not** test whether a composition shift, forced with large weights,
  would change docking — the top-M ceiling makes that unlikely to matter, but it is untested.
- **N = 2 targets** at the loop level (a small sample); the loop-level null leans on the mechanism
  diagnostic (per-motif, ceiling) rather than on target count.
- The null is specific to **promote-only soft reweighting** of building-block/template *selection* at
  `temperature_reactant = 0.1`. Two untried levers remain open as future work: raising the sampling
  temperature so the bias has room to act, and a mechanism that doesn't rely on a global logit bias —
  fragment/scaffold seeding (growing analogs of the single best docker), a deferred non-goal.
- Docking (smina) is a rigid shape-fit proxy; but the result is a *no-difference between arms* under the
  same scorer, so the proxy's absolute accuracy does not affect the comparison.

## Reproduce

- Shakedown loop (box, `SMINA=$(pwd)/smina.static`, detached):
  ```
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m scripts.optimize_loop \
    --ckpt <SP-C ckpt> --targets data/dock/powered_targets.json --limit-targets 2 \
    --arms enrich,uniform --rounds 3 --budget 100 --n 1000 --k 25 --final-m 10 \
    --out-dir data/dock/sp_l_shakedown
  ```
  → `data/dock/sp_l_shakedown/loop_summary.csv` (the table above);
  per-round `candidates.jsonl` / `dock_scores.csv` / `weights_next.json` under `<target>/<arm>/round_<r>/`.
- Mechanism diagnostic: compare promoted-motif frequency in `enrich` vs `uniform` `round_1/candidates.jsonl`
  against `round_0/weights_next.json`; bite-test regenerates with weights scaled 30×.

Artifacts: `data/dock/sp_l_shakedown/` (NFS share). Code: `synformer/molopt/enrich.py`,
`scripts/generate_enriched.py`, `scripts/optimize_loop.py`, hook in `synformer/models/synformer.py`.
