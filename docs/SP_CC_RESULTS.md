# SP-CC: Candidate-Regime Scorer-Agreement Benchmark — Results

**Date:** 2026-07-11 · Branch `sp-cc-candidate` · Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-11-candidate-regime-scorer-agreement*.md`

## The question

SP-CS found score-averaging consensus doesn't beat Boltz on **known-vs-random** discrimination, and
conjectured (untested) that smina's failure mode is specific to the **candidate/optimization** regime.
This benchmark set out to test that: *do smina and Boltz disagree more on generated candidates than on
real known/random molecules?*

**Answer, after controlling for a confound: no — the apparent contrast was an artifact. But the two
scorers do select largely different candidate top-molecules, which is the finding that matters for
selection.**

## Method

Per target, a deterministic stratified sample of ~30 SP-C pocket candidates across the full smina range
(`dock_scores_pocket.csv`) was co-folded with Boltz-2 (150/150 cells, 0 failures). Metrics: per-target
**Spearman** between smina and Boltz strength (`strength = −score`) in the candidate regime vs the
known/random regime; and, on candidates, the **overlap of smina-top-5 vs Boltz-top-5** picks. 5 targets
(P06537's known/random side unavailable — SP-CS dropped it at 3 knowns).

## The confound — and the control that exposes it

Naively, candidate-regime agreement looked much lower than known/random:

| | smina↔Boltz Spearman (mean) |
|---|---|
| candidate regime | **+0.194** |
| known/random regime (full set) | +0.717 |

But the known/random set is **bimodal by construction** — real drugs (strong by both scorers) vs random
decoys (weak by both) — so its +0.72 mostly measures the *easy, coarse* known-above-random split baked
into the set, not fine-grained rank agreement. The candidate set has no such class structure. The honest
control is **within-class** Spearman (range-matched), pooled over the 4 dual-regime targets:

| set | smina↔Boltz Spearman (mean) |
|---|---|
| known/random, **full (bimodal)** | +0.717 |
| known/random, **knowns-only** | **+0.268** |
| known/random, **randoms-only** | **+0.238** |
| **candidate regime** | **+0.194** |

**Within class, known/random agreement (0.24–0.27) is essentially the same as candidate agreement
(0.19).** The +0.72 was the bimodality artifact. So there is **no evidence that scorer disagreement is
optimization-specific** — smina and Boltz have only **modest fine-grained rank agreement (~0.2–0.27) in
both regimes.** The originally-hypothesised "candidate regime is special" headline is **retired.**

## The clean, non-circular finding — scorers pick different candidates

The soundest result is symmetric and references neither scorer as truth: within each target's candidate
pool, **smina's top-5 and Boltz's top-5 are largely different molecules.**

| target | Jaccard(smina-top5, Boltz-top5) |
|--------|--------------------------------|
| O43570_WT | 0.00 |
| P02753_WT | 0.11 |
| P06537_WT | 0.11 |
| P10721_WT | 0.25 |
| P0C559_WT | 0.43 |

Overlap is ≤0.25 on 4/5 targets (0.00 on O43570). **Which scorer you select candidates with materially
changes the molecules chosen.** Combined with the modest fine-ranking agreement above, this says the two
scorers genuinely diverge on *ranking generated candidates* — not more than they diverge on real
molecules, but enough that scorer choice is consequential for selection.

## Verdict

- **Retired:** "smina and Boltz disagree more on candidates than on known/random." That contrast was a
  bimodality artifact; within-class the regimes are indistinguishable (~0.2 vs ~0.25).
- **Supported:** smina↔Boltz fine-ranking agreement is **modest everywhere (~0.2–0.27)**, and on
  candidates the two scorers **select largely different top-molecules (Jaccard ≤0.25).**
- **Bottom line for selection:** scorer choice matters for *which* candidates you pick, in any regime —
  but this benchmark does **not** show the candidate regime is special, and it does **not** establish
  which scorer is right. The recommendation to **validate generated candidates with Boltz** rests on
  SP-CS's *independent* evidence (Boltz AUROC 0.95 ≥ smina on known/random), not on this benchmark.

## Caveats

- Correcting the confound (within-class control) was the key step; the naive cross-regime Spearman should
  not be cited.
- N = 5 targets (4 with both regimes), 30 candidates/target — directional; per-target within-class ρ is
  noisy (e.g. P02753 knowns-only −0.06).
- Candidate pools are smina-biased (SP-C generates smina-decent molecules); the Jaccard/top-5 divergence
  is over a restricted-quality set.
- Hacking framing dropped: on the raw SP-C pool smina-top candidates are not systematically Boltz-weak
  (only O43570 leaned that way); acute smina-hacking was an SP-F *hill-climbing* phenomenon, not a
  property of the raw pool.

## Reproduce

- Candidate Boltz (proxy required; `BOLTZ=.venv-boltz/bin/boltz`):
  `env https_proxy=… .venv-boltz/bin/python -m scripts.candidate_boltz --targets <5> --n 30 --scores data/dock/sp_cc_candidate_boltz.csv`
- Agreement + overlap: `.venv/bin/python -m scripts.candidate_agreement --candidate-boltz data/dock/sp_cc_candidate_boltz.csv --kr-boltz data/dock/sp_cs_boltz_controls.csv`
- Within-class control: per target, `spearmanr(−smina, −boltz)` on knowns-only and randoms-only from
  `dock_scores.csv` ⋈ `sp_cs_boltz_controls.csv` (the numbers in the confound table).

Artifacts: `data/dock/sp_cc_candidate_boltz.csv`, `scripts/candidate_boltz.py`, `scripts/candidate_agreement.py`.
