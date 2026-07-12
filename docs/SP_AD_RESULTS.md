# SP-AD: ADMET Profile of Generated Molecules — Results

**Date:** 2026-07-12 · Branch `sp-ad-admet` · Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-12-admet-harness*.md`

## The question

How drug-like/safe are the molecules the pocket-conditioned SynFormer actually generates — beyond the
crude QED/SA gates used so far — and how restrictive would an ADMET guard be for the per-molecule
specificity DPO?

## Method

`admet-ai` (ML-ADMET, ~40 endpoints; isolated `.venv-admet`) scored **5,701 unique SMILES** pooled and
deduped across the 41 SP-SC pocket-candidate files (`scripts/admet_score.py`). `admet_pass` = raw
classifier probs for **hERG, DILI, ClinTox, Carcinogens < 0.5** AND **HIA ≥ 0.5** (directionality verified:
hERG 0.008 ibuprofen vs 0.976 terfenadine).

## Result — synthesizable and QED-reasonable, but ADMET-liable

**Only 5.4% of generated molecules pass the safety+absorption guard.** The failure is concentrated:

| endpoint | favorable (raw < 0.5) | median drugbank-approved percentile | read |
|---|---|---|---|
| **DILI** | **17%** | p80 | most flagged for liver-injury risk; worse than 80% of approved drugs |
| **hERG** | 41% | p68 | majority predicted hERG-risky |
| ClinTox | 94% | p51 | fine |
| Carcinogens | 84% | p63 | fine |
| CYP1A2/2C19/2C9 | — | p87–92 | high metabolic interaction |
| **Solubility (AqSolDB)** | — | **p25** | poor aqueous solubility |
| HIA (absorption) | — | p81 | good |
| QED / Lipinski / Bioavailability | — | p69 / p64 / p70 | drug-like on physchem |

**Interpretation:** the generator produces **synthesizable, physchem-/QED-reasonable** molecules that
nonetheless carry **pervasive predicted safety and metabolic liabilities** — DILI and hERG above all,
plus high CYP interaction and poor solubility. The QED/SA/heavy-atom gates used earlier in the project
therefore **badly overestimate drug-likeness**: real (ML-predicted) ADMET is far worse than those proxies
implied. This is a genuinely useful, somewhat sobering characterization of what the pipeline generates.

## Implications for the per-molecule specificity DPO

- A **5.4% pass rate means the ADMET guard is very restrictive** — gating the DPO reward on full
  `admet_pass` would reject ~95% of candidates. Options for the DPO: (a) add ADMET-cleanliness as an
  explicit *objective* (push the generator toward the rare drug-like 5.4%) — valuable but harder;
  (b) relax the guard to a **binding-relevant subset** (e.g. hERG + solubility only) so it filters gross
  liabilities without collapsing the pool; (c) treat `admet_pass` as a *readout* rather than a hard gate.
  This distribution is the data needed to choose sensible DPO thresholds.

## Caveats

- **These are ML predictions (proxies), not measured ADMET** — directional triage, not ground truth.
  **DILI models are notoriously noisy / prone to over-flagging** certain chemotypes, so the 83% DILI-fail
  should be read as "this chemotype trips the DILI model," not a hard verdict.
- The `admet_pass` thresholds (0.5) are conservative defaults; the AND of four toxicity criteria + HIA
  compounds, which is why the pass rate is low. Per-endpoint favorable fractions (above) are the more
  informative view.
- Pool = the 41 SP-SC targets' pocket candidates (SP-C model); not the full REAL-space chemistry.

## Reproduce

```
.venv-admet/bin/python -m scripts.admet_score --pools data/dock/candidates_pocket \
  --out data/dock/admet_candidates.csv --summary data/dock/admet_candidates_profile.json
```
(invoke as `-m`; a bare path hits an `optuna`/`hyperopt` import shadow via `scripts/hyperopt.py`.)

Artifacts: `data/dock/admet_candidates.csv` (per-molecule, ~40 endpoints + `admet_pass`),
`admet_candidates_profile.json`; harness `scripts/admet_score.py`.
