# Docking-Selection POC — Results & Verdict

**Date:** 2026-07-06 · **Branch:** `docking-selection` · **Spec:** `docs/superpowers/specs/2026-07-06-target-specific-docking-selection-design.md`

## Bottom line

The reproduction (`FINDINGS.md`) established that the protein-conditioned SynFormer makes valid,
novel, synthesizable molecules but is **not demonstrably target-specific** — three independent
controls on the base model (ligand similarity, proxy affinity, model likelihood) were null or weak.
This POC adds the missing binding signal by **generate → dock → select**: generate synthesizable
candidates from our best generator (SP2 `masked`), dock them into real holo pockets with smina, and
select the best.

Two questions, two answers:

1. **Does docking selection surface good binders?** — **Yes, decisively.** The top-10 selected
   candidates dock better than random Enamine blocks in **all 5** targets (by 2.5–4 kcal/mol) and
   **match or beat the targets' known ligands** in all 5.
2. **Are the selected molecules target-*specific* (mismatch control)?** — **Yes, modestly — and this
   is the first positive specificity signal in the whole project.** A target's docking-selected
   molecules preferentially fit their own pocket. The raw effect is small and confounded by
   per-pocket "dockability"; after correctly removing that confound (per-pocket normalization) the
   signal is clearer: **win-rate 4/5, mean own-pocket rank 2.2 of 5 (chance 3.0).**

The targeting comes from **selecting against the pocket**, exactly as the thesis predicted — not from
the sequence conditioning. Everything ran locally on an Apple-Silicon Mac (CPU smina, MPS generation).

---

## Setup

**Targets** (5 diverse families; each an X-ray holo structure ≤2.5 Å with one drug-like co-crystal
ligand; all present in both the ESM-embedding set and the SP2 test split):

| Target | UniProt | Family | PDB | Co-crystal ligand | Known ligands (test set) |
|---|---|---|---|---|---|
| CA12 | O43570 | carbonic anhydrase (metalloenzyme) | 1JD0 | AZM (acetazolamide) | 896 |
| GR | P06537 | nuclear receptor | 3MNP | DEX (dexamethasone) | 3 |
| KIT | P10721 | receptor tyrosine kinase | 1T46 | STI (imatinib) | 277 |
| RBP4 | P02753 | lipocalin transporter | 1BRP | RTL (retinol) | 13 |
| gyrase-B | P0C559 | bacterial ATPase | 6Y8O | NOV (novobiocin) | 8 |

**Candidates:** ~150 unique valid synthesizable SMILES/target, sampled from the SP2 `masked`
checkpoint on MPS. **Docker:** smina (conda-forge, AutoDock Vina 1.1.2 scoring), autobox from the
co-crystal ligand, exhaustiveness 8, seed 42, CPU. **References per target:** up to 30 known ligands
(availability-capped) + 30 random Enamine building blocks. **Total:** 1,159 docking runs, ~50 min
wall-clock. Scores in kcal/mol, **lower = stronger**. Pipeline is crash-safe (per-(molecule,pocket)
incremental append + resume).

---

## Result 1 — Selection works (selection-vs-random, selected-vs-known)

Own-pocket docking, recomputed from the raw per-dock scores:

| Target | top-10 selected | all candidates | known ligands | random REAL |
|---|---|---|---|---|
| CA12    |  **−8.35** | −5.96 | −6.96 | −5.63 |
| GR      |  **−8.90** | −5.31 | −3.53¹ | −5.74 |
| KIT     | **−11.25** | −9.08 | −10.60 | −7.24 |
| RBP4    | **−10.33** | −8.09 | −8.79 | −6.83 |
| gyrase-B|  **−8.68** | −7.29 | −7.40 | −5.94 |

¹ GR known-ligand mean is over only 3 test ligands.

- **Selected > random in 5/5** (by 2.5–4.0 kcal/mol) — docking selection strongly enriches for binders.
- **Selected ≥ known in 5/5** — the selected generated molecules dock as well as or better than the
  target's *actual* known ligands (including beating imatinib-era KIT ligands and the acetazolamide
  reference for CA12). The generator supplies a broad synthesizable pool; docking does the ranking.
- The *average* candidate is mediocre (worse than known in 4/5) — selection, not raw generation, is
  what delivers quality. This is the intended division of labour.

## Result 2 — Target-specificity (the mismatch control)

Each target's **top-10 selected** candidates were docked into **every** target's pocket. Matrix entry
M[i,j] = best (min) score of target *i*'s selected molecules in pocket *j*. Diagonal = own pocket.

```
                pocket→   CA12     GR     KIT    RBP4  gyraseB
 selected-for CA12       -8.70  -8.70  -11.70  -10.60   -8.90
 selected-for GR         -8.40  -9.20  -10.50   -9.90   -8.20
 selected-for KIT        -7.50  -8.40  -11.60  -10.80   -8.40
 selected-for RBP4       -8.20  -9.40  -12.00  -10.50   -9.60
 selected-for gyrase-B   -8.40  -8.00  -11.80  -10.30   -9.80
```

**Raw mismatch summary** (`mismatch_summary`): own_mean −9.96, offdiag_mean −9.49,
**delta −0.475** (own binds better), **win-rate 3/5**.

### The confound — and the honest metric

The raw numbers *understate* specificity because docking scores are dominated by **which pocket is
easy**, not by molecule–pocket match. Read the matrix by **column**: the KIT pocket scores *everyone's*
molecules −10.5 to −12.0 and RBP4 −9.9 to −10.8, while CA12/GR/gyrase-B pockets score everyone
−7.5 to −9.4. So the raw off-diagonal mean is dragged down by a couple of promiscuous pockets, and the
aggregate delta is carried almost entirely by KIT (per-target delta −2.82) — an artifact of KIT being
an easy pocket, not evidence that KIT's molecules are KIT-specific.

Removing the per-pocket dockability bias by **z-scoring within each pocket (column)** gives the metric
that actually answers "does a target's molecules fit *its own* pocket better than other molecules do":

| | own (diag, z) | off-diag (z) | delta | own-in-pocket rank (of 5) |
|---|---|---|---|---|
| CA12    | −1.14 | −0.18 | **−0.96** | **#1** |
| GR      | −0.90 | +1.12 | **−2.02** | #2 |
| KIT     | −0.15 | +0.54 | **−0.70** | #4 |
| RBP4    | −0.26 | −0.77 | +0.51 | #3 |
| gyrase-B| −1.29 | +0.23 | **−1.52** | **#1** |

**Normalized: delta −0.937, win-rate 4/5, mean own-pocket rank 2.20 of 5 (chance 3.0).**

The picture inverts informatively: KIT — which dominated the *raw* signal — ranks its own molecules
only #4/5 once normalized (its pocket just binds everything). The **genuine** specificity is in CA12
and gyrase-B (both #1), where the docking-selected molecules are the best-fitting of all five sources
into their own pocket. Only RBP4 (a promiscuous hydrophobic lipocalin) fails to prefer its own.

**This is the first positive, direction-consistent target-specificity result in the project** — and it
appears precisely where the thesis said it would: from selection against the 3D pocket, not from the
generator, whose pooled candidates remain generic.

---

## Caveats (honest)

- **Small, single-structure sample:** 5 targets, one holo PDB each, one docking seed, one docker
  (smina; fast but an approximate empirical scoring function — not experimental affinity).
- **Modest absolute effect:** raw own-vs-mismatch is ~0.5 kcal/mol (within docking noise); the
  robust statement is the *normalized/rank* one (own preferentially fits own pocket in 4/5), not a
  large energy gap.
- **Pocket-dockability confound is real** and must be normalized out — the naive `delta`/`win_rate`
  from `mismatch_summary` (3/5) is confounded and should be reported alongside the normalized view,
  never alone.
- **Specificity is bounded by the candidate pool:** docking can only pick from what the generator
  produced; a genuinely pocket-complementary scaffold absent from the pool can't be selected. A
  larger pool or pocket-conditioned generation is the next lever.
- **Reference means in `dock_select_summary.csv`** are slightly affected by a known de-dup quirk
  (identical-valued reference scores collapsed); all numbers in this doc were recomputed from the raw
  `dock_scores.csv`, which is authoritative.

## What this means

Docking selection converts the base model's honest weakness (no demonstrable targeting) into a
working, deployable improvement: **the pipeline reliably surfaces synthesizable molecules that dock as
well as known drugs and preferentially fit their intended pocket.** It validates the "improve the
existing model by selecting against the pocket" path over sequence-only conditioning, using only
open, CPU/Apple-Silicon-runnable tools. Natural next steps: more targets/seeds with confidence
intervals, an orthogonal co-folding spot-check (Boltz-2) on the very top hits, and — if pursued —
a pocket-conditioned generator so targeting enters generation, not just selection.

## Artifacts
- Scores: `data/dock/dock_scores.csv` · summaries: `data/dock/dock_select_summary.csv`,
  `data/dock/dock_mismatch_summary.csv` (all gitignored data)
- Targets: `data/dock/targets.json` · candidates: `data/dock/candidates/<target>.txt`
- Code: `synformer/dock/` (geometry, receptor, dock) · `scripts/dock_prepare.py`, `scripts/dock_select.py`
- Setup: `docs/DOCKING_SETUP.md`
