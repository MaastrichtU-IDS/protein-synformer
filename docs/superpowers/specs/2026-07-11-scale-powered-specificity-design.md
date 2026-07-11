# SP-SC: Scale the Powered Specificity Study — Design

**Date:** 2026-07-11 · **Sub-project:** Priority 3 (scale) from the original handoff.
· **Depends on:** the powered docking-selection harness (`powered_run`/`powered_analyze`), the SP-C
pocket model + prebuilt pockets, and the drug-like-holo test pool.

## 1. Motivation & goal

The powered docking-selection specificity study (own-vs-mismatch normalized delta) is the project's one
positive result but was N=20. This scales it to the **full available drug-like-holo test set** to tighten
CIs and test whether the modest specificity signal holds at larger N.

> Recompute the own-vs-mismatch **normalized delta** (per-pocket z; more negative = more specific) over
> **N=67 targets** with **sampled mismatch** (each source's top-M docked into its own pocket + K random
> mismatch pockets), and compare to the N=20 result.

## 2. Targets

- **Ceiling:** 76 accessions in `druglike_holo_accs.json` ∩ the sp2 test split.
- **67 have a prebuilt `<acc>_WT` pocket** (required for pocket-candidate generation); **9 do not** and
  are **excluded** (building their pockets is a possible later add-on, out of scope here).
- **N=67** = the 20 current targets (reuse `candidates_pocket/`) + **47 new** pocket-ready targets.
- Target set produced by extending `powered_targets` selection to the pocket-ready 67 (drug-like single
  holo ligand vetted per existing criterion) → `data/dock/powered_targets_67.json`.

## 3. Why sampled mismatch (not full all-pairs)

Docking is **CPU-bound** (smina ~7 cores ⇒ ~4 concurrent, ~240 docks/hr; GPUs don't help). Full all-pairs
is **quadratic**: N=67 ⇒ 67×10×67 ≈ 45k docks ≈ 8 days — infeasible. **Sampled mismatch** docks each
source's top-M into its **own pocket + K=12 seeded-random mismatch pockets** ⇒ 67×10×13 ≈ **8.7k docks
≈ 36h** — feasible, and still estimates the per-pocket-normalized delta.

## 4. Method

- **Candidates:** pocket model (SP-C ckpt), `dock_prepare generate-pocket`, for the 47 new targets
  (GPU; existing 20 reuse `candidates_pocket/`). Sequence candidates are **not** available (full
  embeddings absent from the box) — pocket-only, which matches the SP-C arm.
- **Receptors:** `prepare_target` (holo from RCSB via proxy) for the 47 new (existing 20 cached).
- **Docking:** `powered_run` with a **new `--mismatch-sample K` option** — dock each source's top-M into
  its own pocket + K pockets sampled uniformly at random (seeded) from the others; skip the full-all-pairs
  loop. Sharded via `--sources` for parallelism (4 concurrent). `SMINA` env set.
- **Analysis:** `powered_analyze` **unchanged** — `_delta_win_from_matrix` builds the (now sparse) matrix
  and normalizes per pocket-column with **nan-aware** `np.nanmean/nanstd`, so the sampled matrix (own +
  ~K finite entries/column) is handled as-is. delta_i = z(own) − mean(z(sampled mismatch)); bootstrap CI
  over the 67 targets.

## 5. Readout

Own-vs-mismatch normalized delta + win-rate + bootstrap 95% CI at N=67, **vs the N=20 baseline**
(crystal delta −0.62 / −0.71). Does the modest specificity signal survive/tighten at 3× the targets?
1 seed (matching the powered study; multi-seed is a follow-on).

## 6. Components

| Piece | New/reuse | Notes |
|---|---|---|
| target selection → `powered_targets_67.json` | reuse `powered_targets` (extend N, restrict to pocket-ready) | vets drug-like holo; network PDB fetches for the 47 new |
| `powered_run --mismatch-sample K` | **small new (TDD)** | dock own + K seeded-random mismatch pockets instead of all |
| pocket candidate generation (47 new) | reuse `dock_prepare generate-pocket` | GPU |
| receptor prep (47 new) | reuse `prepare_target` | RCSB via proxy |
| `powered_analyze` | reuse **unchanged** (nan-aware per-column z) | |
| `docs/POWERED_SCALE_RESULTS.md` | new | |

Artifacts on the share: `data/dock/powered_targets_67.json`, `candidates_pocket/<new>.txt`,
`dock_scores_scale.csv`, results doc.

## 7. Scope & compute

~8.7k docks ≈ 36h (sharded 4-wide) + candidate gen (GPU, ~hours for 47) + receptor prep (network).
~1.5–2 days wall. 1 seed. Crystal arm only (AF arm is a further follow-on).

## 8. Error handling / caveats

- **9/76 excluded** (no pocket); N=67 is "pocket-ready full set", not literally 76.
- **Sampled mismatch** ⇒ per-pocket z columns have ~13 finite entries (vs 67 in full all-pairs) — noisier
  normalization; report per-column finite counts, and use a **fixed seed** for the mismatch sampling so
  it's reproducible. A pocket sampled by very few sources gives an unstable column — flagged, not fatal.
- **Pocket candidates only** (no sequence arm) — consistent with SP-C; not the SP2 sequence study.
- Receptor prep / candidate gen for 47 new targets may hit individual PDB failures (multi-chain, missing
  ligand) — skip + log per target (as `powered_run`/`af_receptor` already do); final N may be <67.
- 1 seed — separates neither generation nor sampling stochasticity; multi-seed is a follow-on.

## 9. Testing (TDD)

- `powered_run` mismatch-sampling: the pocket-sampler picks own + exactly K distinct others, seeded
  (deterministic), never samples own twice; K≥#pockets ⇒ all pockets (degenerates to all-pairs).

## 10. Non-goals

- No AF arm (crystal only this pass).
- No sequence candidates (embeddings absent).
- No pocket-building for the 9 missing (excluded).
- No multi-seed (follow-on).
