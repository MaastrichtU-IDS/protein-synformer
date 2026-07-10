# SP-CS: Consensus-Scorer Discrimination Benchmark — Design

**Date:** 2026-07-10 · **Sub-project:** consensus selection scorer, following the SP-L/SP-F finding
that single docking/co-folding proxies disagree. · **Depends on:** existing smina known/random scores,
the `boltz_controls` co-folding machinery, and the known/random ground truth in `dock_scores.csv`.

## 1. Motivation & claim

SP-L and SP-F converged on: **selection against a single rigid proxy is not enough** — smina and Boltz
disagree (smina-hacking; direction flips on 3/5 SP-F targets). The evidence-driven next lever is a
**consensus scorer**. This sub-project tests the cleanest ground-truthed version of the question:

> Does a **consensus of smina + Boltz** separate **known binders from random decoys** better — and, above
> all, **more robustly** — than either scorer alone?

The hypothesis is not that consensus beats the best single scorer's *mean* (Boltz alone is likely
near-ceiling), but that consensus **rescues the targets where any single scorer fails** — the exact
failure mode SP-F exposed (smina catastrophically mis-ranks on the P10721-type greasy-basin target).

### Readouts (both, robustness-led)
- **Primary:** per-target **worst-case AUROC** (min across targets) for smina-alone, Boltz-alone,
  consensus. **Win = consensus worst-case > each single scorer's worst-case.**
- **Secondary:** **mean AUROC** across targets (parity-or-better expected, not the headline).

## 2. Data & ground truth

- **smina:** `data/dock/dock_scores.csv` — own-pocket (`pocket == target`) rows with `source ∈
  {known, random}` (530 known / 600 random total). `score` = kcal/mol, lower = stronger.
- **Boltz:** generated via `scripts.boltz_controls` (co-folds each target's known+random into its own
  sequence). `affinity_pred`, lower = stronger. **Proxy required for the MSA server** (the SP-F lesson).
- **Join** smina ⋈ Boltz on `(target, molecule)`; label `known` = positive, `random` = negative.
- Ground-truth caveat: "random" REAL decoys may include incidental weak binders — standard for this
  kind of benchmark; it adds noise, not bias, to all three scorers equally.

## 3. Consensus method

Within each target, over that target's known∪random molecules:
- **Rank-mean (Borda) — primary:** rank molecules by smina strength and by Boltz strength; consensus
  score = mean of the two ranks. Scale-free (immune to the kcal/mol vs affinity-unit mismatch).
- **Z-sum — variant:** z-normalize each scorer's strength within the target; sum. Reported to check the
  result isn't an artifact of the rank transform.

"Strength" = `−score` for both (so higher = stronger, consistent ranking direction).

## 4. Metric

Per target, `AUROC(y_true = is_known, y_score = strength)` for each of {smina, boltz, consensus_rankmean,
consensus_zsum}. Aggregate: **mean** and **min (worst-case)** across the 4 usable targets. A target with
<5 knowns (or <2 of either class) is skipped (unstable/undefined AUROC) and logged — this drops P06537.

## 5. Scope (shakedown-first)

- **Candidate targets (SP-F set) and their available knowns** (own-pocket, from `dock_scores.csv`):
  O43570 30, P10721 30, P02753 13, P0C559 8, **P06537 only 3**. AUROC with 3 positives is unstable.
- **Require ≥5 knowns per target ⇒ drop P06537** → **4 usable targets** (O43570, P10721, P02753, P0C559).
- **Cap 10 known + 10 random / target** (fewer where unavailable — P0C559 uses 8 known). ≈ **~70–80
  Boltz cells** (~few hours; smina is free). Deterministic: first N unique per (target, class) in CSV
  order (as `boltz_controls --cap` already does).
- The worst-case over 4 targets is thin, and P0C559 (8 knowns) is low-N; both are stated caveats. Scaling
  to more targets/molecules is an explicit follow-on if the signal warrants.

## 6. Components

| Piece | New/reuse | Notes |
|---|---|---|
| 5-target Boltz inputs JSON (subset of `matrix_inputs_powered.json`) | **new (tiny)** | so `boltz_controls` enumerates only the 5 targets' known/random |
| `scripts.boltz_controls` run (`--cap 10 --batch --no-kernels`, proxy set) | reuse | generates Boltz known/random → scores CSV |
| `scripts/consensus_score.py` — join smina+Boltz, rank/z consensus, per-target AUROC, mean+min | **new (TDD)** | pure analysis; `roc_auc_score` from sklearn |
| smina known/random | reuse `dock_scores.csv` |
| `docs/SP_CS_RESULTS.md` | new |

Artifacts on the share: Boltz scores CSV (`data/dock/sp_cs_boltz_controls.csv`), consensus report.

## 7. Error handling / caveats

- **sklearn availability:** confirm `sklearn.metrics.roc_auc_score` imports in `.venv` (spike in Task 1);
  if absent, implement a dependency-free AUROC (Mann–Whitney U / rank formula).
- **Join misses:** a molecule scored by only one scorer is dropped from that target's benchmark (logged);
  the AUROC uses the intersection.
- **Boltz nan / MSA failures:** excluded and logged (as in SP-F).
- **Proxy:** the Boltz run MUST have `https_proxy` set (the SP-F gotcha) or all MSA requests fail.

## 8. Testing (TDD)

- `consensus_score.py`: on a hand-built `(target, molecule, class, smina, boltz)` frame — rank-mean and
  z-sum consensus values; per-target AUROC (a constructed case where smina alone mis-ranks one target but
  consensus rescues it, proving the worst-case metric responds); intersection-join drops one-scorer
  molecules; class-too-small target skipped.

## 9. Non-goals (YAGNI)

- No ML/DeepPurpose third scorer (absent; env setup out of scope).
- No in-loop consensus (SP-F showed the loop doesn't help; this is a selection benchmark).
- No full 20-target run in this spec — 5-target shakedown first.
- No new docking (smina known/random already exist).
