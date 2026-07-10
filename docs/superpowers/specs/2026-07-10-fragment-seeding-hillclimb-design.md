# SP-F: Fragment-Seeding Hill-Climb — Design

**Date:** 2026-07-10 · **Sub-project:** generation-side local optimization, following the SP-L null.
· **Depends on:** SynFormer analog sampler, a molecule-encoder checkpoint (`sf_ed_default.ckpt`), the
docking harness, and the SP-L two-arm orchestration/readout machinery.

## 1. Motivation & claim

SP-L showed that **motif-enrichment** of a frozen generator toward docking-winners is a structural
no-op: the winners use the generator's *modal* building blocks/templates, so there is nothing
distinctive to amplify. This sub-project tries the one generation-side lever that does **not** depend on
winners being distinctive: **local search around a specific good binder**.

> Dock a pool → take the top-k binders → generate their **synthesizable neighbors** with SynFormer's
> analog sampler (conditioned on each seed *molecule*) → dock the neighbors → re-seed on the new top-k →
> iterate. Does exploring a good binder's synthesizable neighborhood produce better binders, over a
> docking-budget-matched control?

The pocket enters **only** via docking-selection of what to seed on; generation is conditioned on the
seed molecule, not the pocket — consistent with the project thesis that targeting comes from selection.

### Readouts

- **Primary (science):** budget-matched, per-round **top-M docking** comparison across three arms
  (below), decomposing the effect.
- **Secondary:** best-binder-reached curve; own-vs-mismatch **specificity** (reuse `powered_analyze`);
  per-round **diversity** (hill-climb neighborhood-collapse is a monitored risk, reported not hidden).

## 2. Three budget-matched arms

All arms dock the same number of molecules per round and **share round 0**.

- **treatment** — analog-seed on the current **top-k** dockers; hill-climb (re-seed each round on the
  top-k across everything docked so far).
- **control-A (random-seed)** — analog-seed on **k molecules sampled uniformly at random** from the
  round's docked pool (seeded RNG; same analog mechanism, non-guided seed). Isolates the value of
  **docking-guided** seed choice.
- **control-B (fresh-draw)** — **fresh SP-C pocket-model draws** (the SP-L "uniform" arm; reuse
  `generate_enriched --weights NONE`). Isolates the value of the **analog mechanism** itself.

Decomposition: (treatment − control-A) = docking-guided seeding; (control-A − control-B) = analog
mechanism; (treatment − control-B) = total loop value.

## 3. The loop (per target, R rounds)

```
round 0 (shared): dock B of the existing pocket pool (data/dock/candidates_pocket/<t>.txt) → top-k
for r in 1..R:
    treatment : seeds = top-k over all docked so far
    control-A : seeds = k random/median-docking molecules (seeded RNG)
    control-B : (no seeds — fresh pocket draws)
    for each analog arm: analogs = analog_sample(seeds, sf_ed_default) ; for control-B: fresh pocket draws
    gate (validity + SA ≤ 4 + drug-like) + dedup ; dock B ; record
final_topM (per arm) = best-docked across all rounds
```

`k = 3`, `B = 60`, `R = 2`, `M = 10` (shakedown defaults; §6).

## 4. Model & data

- **Analog sampling:** `data/trained_weights/sf_ed_default.ckpt` (molecule-encoder base model, fetched
  from HF `whgao/synformer`). `fpindex.pkl` + `matrix.pkl` are present in `data/processed/comp_2048/`
  and HF-consistent (same byte sizes). Entry point: `run_parallel_sampling_return_smiles(input=[Molecule…],
  model_path, …)`.
- **control-B** and **round-0 pool** use the existing SP-C artifacts (pocket model + `candidates_pocket`).
- Analog sampler is molecule-conditioned (encodes the seed's atoms/bonds/SMILES) — it **cannot** run on
  the SP-C pocket model; hence the base checkpoint.

### Task-0 feasibility spike — DONE (2026-07-10), passed

Verified on the box GPU before committing to the build:
- **Required fix:** the SP-C pocket refactor commented out `featurize_stack_actions` + `featurize_stack`
  in `synformer/data/common.py` (inside a `'''…'''` block), which the analog sampler imports. Restoring
  just those two functions (self-contained: `TokenType` + `fpindex` + `Stack`, all present; leave
  `create_data` commented) un-breaks the import. **This is implementation Task 1.**
- `sf_ed_default.ckpt` (encoder_type `smiles`) loads into the current pocket-refactored `Synformer`
  class with **0 missing / 0 unexpected** state-dict keys (the encoder is a config-driven factory that
  still supports the molecule encoder).
- `StatePool` analog sampling of ibuprofen produced **318 valid neighbors** including exact
  reconstruction and close analogs. Mechanism confirmed working.

## 5. Anti-hacking, validation, error handling

- **Gate:** reuse `synformer.molopt.enrich.passes_gate` (RDKit-valid + heavy≥`MIN_HEAVY_ATOMS` + allowed
  elements + SA≤4) on all candidates before docking.
- **Independent validation:** Boltz-2 on the **treatment** arm's final top-M (reuse `boltz_matrix`),
  as in SP-L — the honesty check on any docking win.
- **nan** docks excluded from top-k/seeds and stats; **empty seeds** (all nan) → arm logs and skips the
  round; **idempotent/resumable** per-round artifacts (never re-dock); determinism seeded from
  `(base_seed, target, arm, round)`.
- **Diversity guard:** if the treatment arm's scaffold diversity collapses across rounds, that is a
  reported result (local search narrowed), not a failure.

## 6. Scope — shakedown first (the SP-L lesson)

- **Task 0 spike (feasibility gate): DONE and passed** (see §4) — the analog sampler loads the base
  ckpt and produces valid analogs on the box, after restoring `featurize_stack` (Task 1).
- **Shakedown:** 2 targets (`O43570_WT`, `P06537_WT` — same as SP-L, for comparability), 3 arms, `R=2`,
  `B=60`, `k=3` (~3 h). Confirm the hill-climb moves and whether treatment separates from the controls,
  **then** decide the full run (more targets, R=3, larger B).
- Full run + specificity matrix + Boltz = explicit follow-on, gated on the shakedown.

## 7. Components

| Piece | New/reuse | Notes |
|---|---|---|
| `synformer/data/common.py` — restore `featurize_stack`(_actions) | **small fix (Task 1)** | un-comment 2 self-contained functions the analog sampler needs (verified in Task-0 spike) |
| `scripts/fragment_loop.py` — 3-arm hill-climb orchestrator | **new (TDD)** | seed-select → analog-sample (subprocess, GPU) → gate → dock → re-seed; resumable; loop_summary |
| analog sampler `run_parallel_sampling_return_smiles` | reuse | GPU analog generation from seed molecules |
| `synformer.dock` + parallel docking + `SMINA` | reuse | scorer |
| `enrich.passes_gate` | reuse | drug-like/SA gate |
| SP-L loop_summary / arm methodology / `powered_analyze` specificity readout | reuse | readouts |
| `generate_enriched --weights NONE` | reuse | control-B fresh pocket draws |
| `boltz_matrix` | reuse | final-top-M validation |

Artifacts on the NFS share: `data/dock/sp_f/<target>/<arm>/round_<r>/…` (`candidates.smi`,
`dock_scores.csv`, `seeds.smi`) + top-level `loop_summary.csv`.

## 8. Testing (TDD)

- seed-selection (top-k vs k-random over a scored dict; determinism); round-transition + resumability
  (completed round not re-docked; treatment re-seeds on all-docked top-k; control-B never seeds); gate +
  nan hygiene; the analog-arm vs control-B branching. Docking + analog generation stubbed in unit tests.
- Task-0 analog smoke is an integration check on the box (GPU), not a unit test.

## 9. Non-goals (YAGNI)

- No modification of the analog sampler's internals (reuse as-is).
- No generator weight updates.
- No pocket-conditioned analog sampling (analog is seed-molecule-conditioned by construction).
- No full multi-target run in this spec — shakedown first.
