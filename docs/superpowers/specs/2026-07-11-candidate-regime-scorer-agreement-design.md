# SP-CC: Candidate-Regime Scorer-Agreement Benchmark — Design

**Date:** 2026-07-11 · **Sub-project:** the candidate-regime follow-on SP-CS left as future work.
· **Depends on:** `dock_scores_pocket.csv` (candidate own-pocket smina), the `boltz_controls` machinery,
and SP-CS's known/random smina+Boltz for the regime comparison.

## 1. Motivation & claim

SP-CS showed score-averaging consensus doesn't beat Boltz on **known-vs-random** discrimination — but
that regime never triggers smina's failure mode (smina discriminates real drugs fine). SP-F showed smina
*does* fail on **optimized candidates** (P10721: smina −14 kcal/mol, Boltz-refuted). This benchmark
tests the candidate regime directly:

> Do smina and Boltz **disagree far more on generated candidates than on known/random molecules**, and
> are smina's **top candidates Boltz-outliers (hacking)** — i.e. is the candidate regime where an
> independent scorer (Boltz) actually matters for selection?

**Headline (non-circular):** the **regime contrast** — per-target smina↔Boltz rank agreement (Spearman)
in the candidate regime vs the known/random regime. Candidate ≪ known/random ⇒ scorer disagreement is
**optimization-specific**, which is the empirical backing SP-CS's interpretation lacked.

## 2. Data & sample

- **Candidate pool:** `dock_scores_pocket.csv`, candidate own-pocket rows (`pocket==target`,
  `source=='candidate'`) — 150 SP-C pocket candidates/target, smina range ≈ −12 to +16. `score` = smina.
- **Targets (5):** `O43570_WT, P06537_WT, P10721_WT, P02753_WT, P0C559_WT`. (P06537's known/random regime
  comparison is weak — only 3 knowns — but its candidate pool is full; noted.)
- **Stratified sample ~30/target** across the smina range: sort by smina, take 10 from the strongest
  third, 10 from the middle third, 10 from the weakest third (deterministic; avoids the range-restriction
  confound that in-hand top-10-only Boltz would impose). Boltz-score them (~150 cells, ~few hours).
- **Boltz** via the `boltz_controls` batch machinery (co-fold each candidate into its target's sequence;
  **proxy required for the MSA server**; `BOLTZ=.venv-boltz/bin/boltz`, `--no-kernels`, GPU).

## 3. Metrics (per target + aggregate)

- **Regime contrast (headline):** `spearman(smina_strength, boltz_strength)` over the ~30 candidates
  (candidate regime) vs over the known/random set (from SP-CS data), per target. Aggregate: mean/median.
  `strength = −score`.
- **Hacking (illustration):** rank candidates by smina; report the **Boltz percentile of the smina-top-k**
  (k=5) within the sampled pool. Low percentile ⇒ smina's best are Boltz-mediocre (hacking).
- **Selection divergence (illustration):** overlap (Jaccard) of smina-top-k vs Boltz-top-k vs
  consensus(rank-mean)-top-k among the sample.

## 4. Honest framing (circularity guard)

The **regime-contrast Spearman is ground-truth-free and non-circular** — it just measures whether the two
scorers agree, and compares two molecule regimes. That is the headline. The **hacking and
selection-divergence metrics use Boltz as the reference**, which is partly circular (they assume Boltz is
"right"); they are reported as **illustration** of *how* disagreement manifests, not as proof consensus
selects better. The SP-CS conclusion stands: prefer **Boltz-as-validator**, not score-averaging.

## 5. Components

| Piece | New/reuse | Notes |
|---|---|---|
| `scripts/candidate_boltz.py` — stratified-sample candidates + Boltz-score | **new (TDD pure sampler)** | sampler pure/tested; Boltz run reuses `boltz_controls._run_batch` |
| `scripts/candidate_agreement.py` — Spearman (both regimes), hacking percentile, selection overlap | **new (TDD)** | pure analysis; `scipy.stats.spearmanr` |
| `dock_scores_pocket.csv`, SP-CS `sp_cs_boltz_controls.csv` + `dock_scores.csv` known/random | reuse | |
| `docs/SP_CC_RESULTS.md` | new | |

Artifacts on the share: `data/dock/sp_cc_candidate_boltz.csv`, results doc.

## 6. Scope

5 targets, ~30 stratified candidates each ≈ ~150 Boltz cells (~few hours). N=5 with 30/target is a
**directional** read (a powered claim needs more); the regime contrast is the robust part.

## 7. Error handling / caveats

- Confirm `scipy.stats.spearmanr` imports in `.venv` (present — scipy used elsewhere).
- Boltz nan/MSA failures excluded + logged (proxy required — the SP-F/SP-CS gotcha).
- Stratified sampler must be deterministic (fixed selection given a smina-sorted pool).
- Hacking/selection metrics are Boltz-referenced (partly circular) — framed as illustration (§4).
- P06537 known/random regime point is low-N (3 knowns) — flag; its candidate-regime point is fine.

## 8. Testing (TDD)

- sampler: deterministic strong/mid/weak thirds selection over a toy smina-scored pool; count per stratum.
- analysis: `spearmanr` on a constructed pool; hacking percentile (a smina-top-but-Boltz-weak fixture
  yields low percentile); selection-overlap Jaccard on constructed top-k sets; regime-contrast helper
  returns per-target candidate vs known/random Spearman from two frames.

## 9. Non-goals

- No new docking (candidate smina exists in `dock_scores_pocket.csv`).
- No AUROC (candidates unlabeled — that was SP-CS's known/random regime).
- No claim that consensus *selects better* (Boltz-referenced ⇒ circular) — headline is the regime contrast.
- No full-pool Boltz (stratified sample bounds compute).
