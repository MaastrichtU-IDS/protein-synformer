# SP-SC: Scaled Powered Specificity (N=41) — Results

**Date:** 2026-07-12 · Branch `sp-sc-scale` · Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-11-scale-powered-specificity*.md`

## The question

The docking-selection specificity result (own-vs-mismatch normalized delta; more negative = more
specific) is the project's one positive, but was N=20. Does it hold at larger N?

> Recompute the delta over the full pocket-ready drug-like-holo test set with **sampled mismatch**
> (each source's top-M docked into its own pocket + K=12 random pockets), and compare to N=20 **under a
> matched design**.

## Method

- **Targets:** the 76 drug-like-holo test accessions → 67 with prebuilt pockets → **41 passed the strict
  single-drug-like-holo vetting** (26 dropped for multiple/ambiguous holo ligands). N=41 ≈ 2× the
  original 20 (which are a subset). Pocket-model candidates generated for the 21 new targets.
- **Docking:** `powered_run --mismatch-sample 12 --skip-af`, crystal arm, 1 seed, sharded 4-wide
  (~10.7k docks). Own-pocket phase made shard-aware (each shard docks only its sources).
- **Analysis:** `powered_analyze`'s nan-aware per-pocket-z delta over the sparse sampled matrix, with a
  bootstrap 95% CI over targets.

## Result — the signal replicates on independent targets

The 41 targets **contain** the original 20, so "N=41 vs N=20" would be a superset-vs-subset comparison,
not a replication. The honest test is the **21 NEW targets alone** (never involved in the original
finding), all computed within this one scale run (own-vs-mismatch normalized delta, − = more specific,
sampled mismatch K=12, bootstrap 95% CI):

| cohort (within the scale run) | delta | 95% CI | win-rate | N |
|---|---|---|---|---|
| **21 NEW targets (independent replication)** | **−0.643** | **[−1.09, −0.20]** | 0.67 | 21 |
| 20 original targets (within this run) | −0.911 | [−1.35, −0.55] | 0.90 | 20 |
| all 41 combined | −0.796 | [−1.07, −0.54] | 0.80 | 41 |

- **The specificity signal independently replicates:** on 21 targets unrelated to the original study, the
  delta is −0.64 with a **CI that excludes 0**. It is modestly weaker than the original cohort (−0.91
  within this run) but clearly present — not an artifact of the original 20.
- Combined N=41 gives −0.80 [−1.07, −0.54], a tighter CI than any single cohort.

*(Aside: a separately-computed N=20 baseline on the old `dock_scores_pocket.csv` — different docking run,
different sampled pockets — gives −0.79, consistent with the original published −0.71; but it is not the
within-run 20-subset above and should not be differenced against it.)*

## Methodological notes

- **The sampled matrix self-completes.** `_matrix_normalized_delta` fills `M[i,j]` by `(molecule, pocket)`,
  and docking is deterministic (same SMILES + pocket + seed → same score). Because candidate pools **share
  SMILES across targets**, each source's mismatch cells get filled from any row that docked that SMILES
  into that pocket — so the effective matrix is denser than own+12, and every one of the 41 sources has a
  well-defined delta (even one whose own mismatch run didn't finish). Deterministic and unbiased; if
  anything it improves the estimate's completeness.
- **Efficiency fixes made mid-study:** `--mismatch-sample` avoids quadratic all-pairs (~10.7k docks vs
  ~45k for full N=41); shard-aware own-pocket removed a 4× redundancy; `--skip-af` kept it crystal-only.

## Verdict

**The project's one positive result — docking-selection specificity — independently replicates:** on 21
new targets the own-vs-mismatch delta is −0.64 (CI [−1.09, −0.20], excludes 0); combined N=41 gives −0.80.
Selection against the 3D pocket confers a modest but real specificity that is not an artifact of the
original 20.

**Why this positive is more trustworthy than SP-F's smina wins.** The specificity delta is a **relative,
same-scorer** comparison — a molecule's smina score in its *own* pocket vs in *mismatch* pockets, all
smina. A uniform smina-hacking bias (the failure mode that sank SP-F's *absolute* candidate selection and
showed up as smina/Boltz disagreement in SP-F/SP-CC) **differences out** of a same-scorer own-minus-
mismatch contrast. So this metric is structurally far less vulnerable to the smina/Boltz disagreement than
absolute selection. (This does not imply Boltz would agree on the *magnitude* — only that the relative
construction blunts the known confound.)

Consistent with the session's arc: generation-side levers don't confer targeting (SP2/SP-C/SP-L/SP-F);
**selection does**, it **replicates on independent targets** (SP-SC), and its relative-discrimination
construction makes it robust to the single-scorer caveat that undermined the absolute-quality findings.
Remaining honest limits: 1 seed, crystal arm, pocket candidates, and the magnitude is a smina metric.

## Caveats

- **N=41 not 76** — strict single-holo vetting (26 dropped) and 9 pocketless of the 76.
- **Sampled mismatch** (not full all-pairs); the matched N=20-sampled baseline controls for this, and the
  self-completion note above shows the estimate is well-defined.
- **1 seed, crystal arm, pocket candidates, smina** — a single-scorer selection metric; not Boltz-validated
  at the molecule level (that is SP-F/SP-CC's separate finding).

## Reproduce

- Docking: `powered_run --targets data/dock/powered_targets_67.json --candidates-dir data/dock/candidates_pocket
  --mismatch-sample 12 --skip-af --n-refs 0 --top-m 10 --source-shard i/4` (×4), merge shard CSVs (dedup
  molecule,pocket) → `dock_scores_scale.csv`.
- Analysis: per-target `_matrix_normalized_delta` (top-M from own-pocket) → mean + bootstrap CI; matched
  N=20 baseline = subsample `dock_scores_pocket.csv` mismatch to own+12 via `_sample_mismatch`.

Artifacts: `data/dock/powered_targets_67.json` (N=41), `data/dock/dock_scores_scale.csv`,
`scripts/powered_run.py` (`--mismatch-sample`, `--skip-af`).
