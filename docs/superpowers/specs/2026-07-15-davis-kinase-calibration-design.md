# Tier-3 (DAVIS): properly-powered kinase docking-selectivity calibration — Design

**Date:** 2026-07-15 · Sub-project: re-run the kinase docking-selection calibration on the **DAVIS** kinome
panel (dense, low-noise, many pairs) to **firm or break** the Tier-2 within-kinase ρ 0.245 before that
number stands as the capstone verdict. Reuses the Tier-2 docking + correlation machinery. · Depends on:
DAVIS (via PyTDC), our kinase crystal structures in `data/dock/powered_targets_67.json`, `synformer.dock`.

## 1. Goal & firm-or-break framing

> Does docked own-vs-mismatch selectivity track **measured** kinase-paralog selectivity when measured on a
> **dense, low-noise, many-pair** ground-truth panel — or was Tier-2's ρ 0.245 (3 pairs, 320 triples, a
> ChEMBL scrape) a thin-data artifact?

- **FIRM:** ρ holds or strengthens with a tight CI across many pairs → the project's one positive is real
  and properly powered; the capstone verdict stands, strengthened.
- **BREAK:** ρ collapses toward 0 on the dense low-noise data → the Tier-2 signal was thin-data noise; the
  one positive breaks and the capstone must say so.

Either outcome is decisive and honest. This is the highest-value remaining *compute* experiment: more
ground truth for the one lever that worked, not another lever.

## 2. Data

- **DAVIS** (PyTDC `DTI(name='DAVIS')`): 68 drugs × 379 kinases, **fully dense** Kd. Convert to
  **pKd = 9 − log10(Kd[nM])**; DAVIS non-binders (Kd = 10000 nM) become pKd 5.0 (kept — they are real
  "does-not-bind" selectivity information).
- **Overlap with our crystal pockets = 15 kinases.** **Primary set = 13 canonical protein kinases**
  (CAMK4, CDK5, CSNK1A1, CSNK1E, CSNK1G1, DAPK2, DYRK1A, FGFR1, JAK3, KIT, NEK1, PHKG1, STK16);
  **robustness set = all 15** (adds PIK3CD lipid kinase + RIOK1 atypical, whose ATP-site geometry differs).
- Each kinase's dockable structure (pdb_id + `ligand_resname` autobox) comes from
  `data/dock/powered_targets_67.json` — already prepped in prior docking runs.
- DAVIS drug identity is matched to our pockets by **base gene symbol** (strip DAVIS mutation/domain
  annotations, e.g. `ABL1(F317I)` → `ABL1`); where a gene has multiple DAVIS rows (mutants), take the
  **wild-type / median** pKd per (drug, gene).

## 3. Method (reuse Tier-2 machinery)

1. Dock the 68 DAVIS drugs into the 15 kinase crystal pockets (ATP-site autobox; DAVIS drugs are
   ATP-competitive, so the site is appropriate). ~68×15 = **~1,020 docks (~1h)**, sharded/parallel like
   the Tier-2 driver.
2. Per pocket, **z-normalize** the docked score over the 68 drugs (common per-pocket scale).
3. For every kinase pair (A,B) and every drug measured on both (dense → all): **measured ΔpKd = pKd(A) −
   pKd(B)**; **docked Δ = z_A − z_B**. Spearman ρ(measured, −docked) (so + = docking tracks selectivity),
   per-pair and pooled, **compound-clustered bootstrap** CI. Report the **distribution of per-pair ρ**
   (how many of the 78 protein-kinase pairs are positive) — the guard against one pair carrying the pooled
   number (the Tier-2 lesson).
4. Report primary (13 protein kinases, 78 pairs) and robustness (all 15, 105 pairs); head-to-head vs the
   Tier-2 pooled 0.245.

## 4. Decision criteria (pre-committed)

- **FIRM** if pooled protein-kinase ρ CI excludes 0 **and** a clear majority of per-pair ρ are positive
  (directional consistency), with the point estimate in the ballpark of (or above) 0.245.
- **BREAK** if pooled ρ CI includes 0, or per-pair ρ are a coin-flip around 0 (pooled number driven by a
  few pairs).
- **Honesty guards:** report the per-pair distribution, not just the pooled ρ; a dense-panel ρ that is
  *lower* than 0.245 but still >0 means "real but even weaker than the ChEMBL scrape suggested" — state
  the magnitude plainly. Docking remains a weak proxy regardless of significance.

## 5. Components & interfaces

| file | responsibility |
|---|---|
| `scripts/davis_prep.py` | load DAVIS via PyTDC → pKd; map DAVIS genes → our pocket target_ids; write drug SMILES set + `measured_davis.json` ({drug_smiles: {gene: pKd}}) + the kinase→pocket list; **pure `base_gene()` + pKd conversion TDD'd** |
| `scripts/davis_dock_driver.sh` | dock the 68 drugs into the 15 kinase pockets (shard/parallel, reuse `powered_run` pattern) |
| `scripts/davis_analyze.py` | per-pocket z; measured-vs-docked Spearman per-pair + pooled + compound-clustered bootstrap; per-pair ρ distribution; primary (13) + robustness (15) |
| `docs/TIER3_DAVIS_RESULTS.md` | verdict; update FINDINGS §E + CAPSTONE |

## 6. Testing (TDD)

- `base_gene()`: `ABL1(F317I)` → `ABL1`, `CSNK1A1` → `CSNK1A1`, `MAP3K1-domain` → `MAP3K1`.
- pKd conversion: Kd 1 nM → pKd 9; Kd 10000 nM → pKd 5.
- per-(drug,gene) aggregation across DAVIS mutant rows (median).
- selectivity Δ + triple construction: matches `tier2_analyze` sign/z convention (reuse it).

## 7. Non-goals / caveats

- **KIBA** (larger panel) is an optional follow-up; DAVIS's dense 68×15 is already a ~20× power gain over
  Tier-2 and sufficient to firm-or-break.
- Docking into crystal structures with heterogeneous bound ligands (each kinase's own inhibitor defines the
  autobox) — a fixed, documented per-pocket choice, identical for all drugs (differences out in the z).
- **Env note:** PyTDC was installed into the main `.venv` and downgraded numpy→1.26.4 / rdkit→2023.09.6;
  the suite still passes (23/23) but this was sloppy — future heavy deps go in an isolated venv.
- No generation, no oracle — this is calibration of the existing docking metric only.
