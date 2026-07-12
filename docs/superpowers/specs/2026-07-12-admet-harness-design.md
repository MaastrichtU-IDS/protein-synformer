# SP-AD: ADMET Harness on Generated Molecules — Design

**Date:** 2026-07-12 · **Sub-project:** ML-ADMET characterization of generated pools + the anti-hacking
guard for the per-molecule specificity DPO. · **Depends on:** `admet-ai` (installed + smoke-verified in
the isolated `.venv-admet` on the share).

## 1. Motivation & goal

Every drug-likeness check so far has been crude (SA + heavy-atom + element allow-list). A real ML-ADMET
panel (`admet-ai`, ~40 endpoints across absorption/distribution/metabolism/excretion/toxicity + Tox21)
lets us (a) **characterize what the generator actually produces** vs approved-drug percentiles, and
(b) provide the **anti-hacking guard** the per-molecule specificity DPO needs — so the DPO can be rewarded
for molecules that are *specific AND ADMET-clean*, not specific-but-undruggable.

> Score any SMILES pool with `admet-ai` → per-molecule endpoint table + a compact profile + an
> `admet_pass` flag. First characterize the 41-target pocket-candidate pools; reuse as the DPO guard/readout.

## 2. Feasibility — spike DONE (2026-07-12)

`admet-ai` 1.4.0 installed cleanly into `.venv-admet` (torch 2.5 + chemprop 1.6.1) and
`ADMETModel().predict(smiles=[...])` returns a pandas DataFrame with ~40 endpoints, each as raw prediction
+ `<endpoint>_drugbank_approved_percentile`. Endpoints include hERG, DILI, ClinTox, Carcinogens, CYP1A2/
2C19/2C9/2D6/3A4 (inhib + substrate), Clearance, Half-life, HIA, Caco2, PAMPA, Pgp, Bioavailability,
Solubility, Lipophilicity, BBB, VDss, PPBR, LD50, and the Tox21 NR-*/SR-* panels.

## 3. Components

| Piece | New/reuse | Notes |
|---|---|---|
| `scripts/admet_score.py` | **new (TDD)** | CLI: score a SMILES pool → endpoint CSV + profile + `admet_pass`. Runs in `.venv-admet`. |
| `admet-ai` | reuse (installed) | `ADMETModel().predict` — import inside `main()` so pure helpers unit-test in `.venv` |

Pure, unit-testable helpers (no admet-ai/GPU in tests):
- `load_pool(paths) -> list[str]` — read SMILES from candidate files / dirs, dedup.
- `admet_pass(df, cfg) -> pd.Series[bool]` — per-molecule pass/fail from an endpoints DataFrame.
- `profile(df) -> dict` — pool-level summary (median key endpoints, % passing, % favorable per critical
  endpoint).

## 4. The `admet_pass` guard (the DPO gate)

A molecule **passes** iff it is not a red flag on critical safety endpoints **and** has adequate
absorption. Concretely (defaults; configurable, and the implementer must **verify exact column names +
directionality** against a live `predict` output before finalizing):
- **Toxicity, low-risk:** the raw classifier probabilities for **hERG, DILI, ClinTox, Carcinogens** below
  a threshold (default 0.5; these are binary classifiers where higher = more toxic).
- **Absorption, adequate:** **HIA** (human intestinal absorption) probability high, and **aqueous
  solubility** not extreme.
- Reported alongside the interpretable `drugbank_approved_percentile`s.

Prefer **raw probabilities** for the guard (unambiguous directionality) over percentiles; use percentiles
for the human-readable profile. If a raw column's semantics are unclear, the implementer verifies with a
2-molecule probe (a known-safe drug vs a known hERG-blocker) and documents the direction.

## 5. First run — characterize the 41 pools

Score `data/dock/candidates_pocket/<target>.txt` for the 41 SP-SC targets (~41×150 ≈ 6,150 unique-ish
SMILES; dedup across pools). Report: overall `admet_pass` rate, per-critical-endpoint favorable %, and how
the generated pools sit vs approved-drug percentiles. This is the "ADMET on generated molecules" deliverable.

## 6. Scope / compute

`admet-ai` inference is fast (CPU or GPU); ~6k molecules is minutes-to-~1h. `.venv-admet` is isolated
(no risk to `.venv`/`.venv-train`). Runs need the proxy only if models re-download (cached after the spike).

## 7. Caveats

- `admet-ai` endpoints are ML predictions (proxies), not measured ADMET — directional, for triage/guarding.
- Percentile directionality varies per endpoint; the guard uses raw probs with verified direction.
- The guard thresholds are defaults; the DPO can tune them, and the 41-pool distributions (this run) inform
  sensible cutoffs.

## 8. Testing (TDD)

- `load_pool`: reads/dedups SMILES from files + a dir; skips blanks.
- `admet_pass`: on a constructed endpoints frame — a clean molecule passes; a high-hERG / high-DILI /
  low-HIA molecule fails; threshold config respected.
- `profile`: aggregates (pass rate, per-endpoint favorable %) on a small frame.
- The `admet-ai` call itself is an integration smoke on the box (already spike-verified), not a unit test.

## 9. Non-goals

- No new model training; `admet-ai` used as-is.
- No docking (orthogonal to specificity).
- The DPO itself is a separate sub-project; this only provides the guard + characterization.
