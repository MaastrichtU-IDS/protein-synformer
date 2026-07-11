# SP-CC: Candidate-Regime Scorer-Agreement Benchmark — Results

**Date:** 2026-07-11 · Branch `sp-cc-candidate` · Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-11-candidate-regime-scorer-agreement*.md`

## The question

SP-CS found score-averaging consensus doesn't beat Boltz on **known-vs-random** discrimination, and
argued (untested) that smina's failure mode is specific to the **candidate/optimization** regime. This
benchmark tests that directly and non-circularly:

> Do smina and Boltz **disagree more on generated candidates than on real known/random molecules**?

## Method

Per target, a **deterministic stratified sample** of ~30 SP-C pocket candidates across the full smina
range (`dock_scores_pocket.csv`, 150 candidates/target → 10 strong / 10 mid / 10 weak) was co-folded
with Boltz-2 (150/150 cells, 0 failures). Headline metric: per-target **Spearman** between smina strength
and Boltz strength (`strength = −score`) in the **candidate regime** vs the **known/random regime**
(reusing SP-CS's known/random smina+Boltz). 5 targets (P06537's known/random side is unavailable — SP-CS
dropped it at 3 knowns).

## Result — scorers disagree far more on candidates (clean, consistent)

**smina↔Boltz Spearman (higher = agree):**

| target | candidate regime | known/random regime |
|--------|------------------|---------------------|
| O43570_WT | +0.092 | +0.721 |
| P02753_WT | +0.175 | +0.665 |
| P0C559_WT | +0.444 | +0.665 |
| P10721_WT | +0.240 | +0.816 |
| P06537_WT | +0.020 | — (no SP-CS known/random) |
| **mean** | **+0.194** | **+0.717** |

**On every target with both regimes (4/4), smina↔Boltz agreement is far lower on generated candidates
than on real known/random molecules** (mean +0.194 vs +0.717 — roughly a 4× drop). The two scorers agree
well on what *real* molecules are, but **barely agree on what the generator produces.** This is the
non-circular headline, and it is clean and directionally consistent across all targets.

### Illustration (Boltz-referenced — see circularity note)

- **Selection divergence:** smina's top-5 and Boltz's top-5 candidates are largely **different molecules**
  — Jaccard overlap 0.00–0.43 (mostly ≤0.25). Selecting by smina vs Boltz picks different candidates.
- **Hacking:** the mean Boltz percentile of the smina-top-5 is low on O43570 (0.40 — smina's best are
  Boltz-mediocre) but middling-to-decent elsewhere (0.59–0.77). So smina-hacking of the raw SP-C pool is
  present but not universal; it was most acute in SP-F, where the loop actively *hill-climbed* smina.

## Verdict

**Scorer disagreement is optimization-specific: smina and Boltz agree on real known/random molecules
(mean ρ 0.72) but not on generated candidates (mean ρ 0.19).** This is the empirical backing SP-CS's
interpretation lacked, and it completes the consensus-scorer arc:

- Known/random regime (SP-CS): scorers agree; Boltz alone is competent; score-averaging adds nothing.
- Candidate regime (SP-CC): scorers **disagree sharply**, so *which* scorer you select with materially
  changes the molecules chosen — this is precisely where an **independent scorer (Boltz) as validator /
  selector** earns its keep.

Actionable, evidence-based recommendation for future generation work: **select/validate generated
candidates with Boltz (or a Boltz-inclusive consensus), not smina alone** — smina and Boltz agree on real
chemistry but diverge on generated candidates, exactly the molecules a generator produces.

## Circularity note & caveats

- The **headline (regime-contrast Spearman) is non-circular** — it only measures whether two scorers
  agree, comparing two molecule regimes; it makes no assumption about which scorer is "right."
- The **hacking and selection-divergence metrics use Boltz as the reference** (they assume Boltz is the
  better scorer), so they are **illustration** of *how* the disagreement manifests, not proof that
  Boltz-selection is correct. SP-CS's independent evidence (Boltz AUROC 0.95 ≥ smina on known/random) is
  what justifies treating Boltz as the more trustworthy scorer.
- **N = 5 targets** (4 with both regimes), 30 candidates/target — directional, not powered. The
  candidate pools are smina-biased (SP-C generates smina-decent molecules), which if anything *understates*
  the disagreement (the weakest-smina candidates that would most expose divergence are under-represented).
- P06537's known/random regime is unavailable (SP-CS dropped it at 3 knowns); its candidate ρ (+0.02)
  still fits the pattern.

## Reproduce

- Candidate Boltz (proxy required for MSA; `BOLTZ=.venv-boltz/bin/boltz`):
  `env https_proxy=… .venv-boltz/bin/python -m scripts.candidate_boltz --targets <5> --n 30 \
  --scores data/dock/sp_cc_candidate_boltz.csv`
- Analysis: `.venv/bin/python -m scripts.candidate_agreement --candidate-boltz data/dock/sp_cc_candidate_boltz.csv \
  --kr-boltz data/dock/sp_cs_boltz_controls.csv`

Artifacts: `data/dock/sp_cc_candidate_boltz.csv`, `scripts/candidate_boltz.py`, `scripts/candidate_agreement.py`.
