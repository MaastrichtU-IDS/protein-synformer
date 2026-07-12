# SP-DPO: Per-Molecule Specificity DPO — Pilot Design

**Date:** 2026-07-12 · **Sub-project:** the one untested generation-side lever — can *weight updates*
confer targeting where static conditioning (SP-C) and frozen re-biasing (SP-L/SP-F) could not? Run as a
**cheap pilot** first (decide full run from pilot signal). · **Depends on:** SP-C pocket model, the
docking harness, `powered_analyze` delta, `sp_f_boltz`, `admet_score`.

## 1. Goal & pre-committed expectation

> Does DPO fine-tuning of the SP-C pocket model on **per-molecule own-vs-mismatch specificity** preference
> pairs make its *raw samples* target-specific on **held-out** pockets, beyond the base model?

**Pre-committed likely outcome (advisor + accumulated evidence):** the generator *can* be pushed to
internalize the **smina/shape-fit** specificity, but (a) Boltz refutes it (method-dependent, per
BOLTZ_VALIDATION) and (b) the pool is ~95% ADMET-undruggable (SP-AD). So the pilot most likely yields a
capstone-null; it is framed that way up front so a smina-only "win" isn't spun. The pilot exists to see if
there is *any* signal cheaply before a ~7-day full run.

## 2. Data & preference pairs (the per-molecule fidelity choice)

- **Targets:** ~10 train + ~4 held-out, **family-diverse** (mix kinase / GPCR / metabolic — not all same
  family, so pairs & eval aren't dominated by one pocket type). Held-out never seen in training.
- **Per train target:** generate a pool **keeping each molecule's route** (`GenerateResult` route tensors,
  which `get_log_likelihood` consumes), dock ~40 molecules **own + K=12 mismatch** → per-molecule
  specificity = z(own) − mean(z(mismatch)). ~10×40×13 ≈ **5.2k docks (~1 day)**.
- **Preference pairs (conditioned on the target's pocket):** winner = high-specificity molecule, loser =
  low-specificity / promiscuous molecule (both are base-model samples, so their routes exist).

## 3. DPO mechanics

- **Policy** = SP-C model (trainable copy); **reference** = frozen SP-C.
- **Loss** (standard DPO): `-log σ( β[(llπ_w − llref_w) − (llπ_l − llref_l)] )`, where `ll·` =
  `model.get_log_likelihood(code, code_padding_mask, token_types, rxn_indices, reactant_fps,
  token_padding_mask)` on the winner/loser routes, **conditioned on the pair's target pocket** (the `code`
  from `encode(pocket_feat)`). β default 0.1. Reference provides the KL leash implicitly.
- Feasible via the existing route tensors (verified) and the molopt precedent (which already runs
  `get_log_likelihood` on a `GenerateResult`).

## 4. Evaluation (held-out, relative, Boltz + ADMET checked)

Sample the DPO'd model conditioned on **held-out** pockets → dock own + mismatch:
- **Primary:** own-vs-mismatch **specificity delta (family-stratified)**, DPO'd model **vs base SP-C**, on
  held-out targets. Win = DPO raw samples more specific than base.
- **Boltz spot-check** on the DPO'd model's held-out top-M (does co-folding corroborate, or is it the
  smina artifact?).
- **ADMET readout** (`admet_score`, soft — not a training gate here): does DPO change druggability
  (`admet_pass` rate) — a check that DPO isn't trading specificity for worse ADMET.

## 5. Components

| Piece | New/reuse | Notes |
|---|---|---|
| generation-with-routes | **new/extend** (`generate_enriched` keeps routes) | need per-molecule route tensors, not just SMILES |
| pair builder (dock own+mismatch → specificity → winner/loser) | **new (TDD pure specificity/pairing)** | ~5.2k docks |
| `scripts/dpo_train.py` — DPO loss + train loop on SP-C | **new (TDD loss)** | the substantial code; reference frozen, policy trained; conditioned |
| held-out eval | reuse `powered_analyze` delta + `sp_f_boltz` + `admet_score` | |
| `docs/SP_DPO_RESULTS.md` | new | |

## 6. Scope / compute (PILOT)

~5.2k pair docks + eval docks (held-out samples own+mismatch) ≈ **~1–1.5 days** docking; DPO training GPU
hours. **Decision gate:** if the pilot shows *any* held-out specificity gain (DPO > base) that survives a
Boltz spot-check, scale to the full ~7-day run; otherwise it's the capstone-null. The DPO *code* is the
same effort at pilot or full scale — the pilot only bounds the *compute/science* risk.

## 7. Caveats / risks

- **Likely-null, pre-committed** (§1).
- **DPO code is the substantial new piece** (route-conditioned loss + frozen reference); pilot doesn't
  reduce that effort.
- Training toward smina specificity = training the **shape-fit signal Boltz refutes** — the Boltz eval is
  the honesty gate.
- Held-out family diversity matters (small N; a kinase-only held-out set would confound with the family
  cross-reactivity).
- 1 seed pilot; DPO can be unstable — monitor for reward collapse / degeneracy (KL to reference guards).

## 8. Testing (TDD)

- per-molecule specificity + winner/loser pairing (pure, on a constructed docked frame).
- DPO loss: on toy log-likelihoods, verify the loss decreases when policy raises winner over loser
  relative to reference, and `β=0` / equal-ll degenerate cases.
- generation-with-routes: routes captured are the tensors `get_log_likelihood` accepts (shape/dtype).

## 9. Non-goals

- No full ~7-day run in the pilot (gated on pilot signal).
- No ADMET *training* gate (soft readout only; SP-AD showed full gate = 5.4%, too strict).
- No REINVENT-online (DPO chosen for oracle-efficiency).
