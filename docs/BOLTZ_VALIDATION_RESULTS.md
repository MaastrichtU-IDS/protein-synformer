# Boltz-2 Co-Folding Mismatch Validation — Results & Verdict

**Date:** 2026-07-06 · **Branch:** `boltz-validation` · **Spec:** `docs/superpowers/specs/2026-07-06-boltz2-cofolding-mismatch-validation-design.md`

## Bottom line

The docking-selection POC (`DOCKING_SELECTION_RESULTS.md`) found the project's first positive
target-specificity signal: docking-selected molecules preferentially fit their own pocket. That
rested on **one** method — smina docking into rigid crystal structures. This sub-project tested it
with an **independent, sequence-only** method: Boltz-2 co-folding, which predicts the protein–ligand
complex de novo from sequence + SMILES and never sees our crystal structures.

**Verdict: Boltz-2 does NOT corroborate the docking target-specificity.** Across all three Boltz
metrics the selected molecules show ~no preference for their own pocket, the folds are
high-confidence (so it is not a folding failure), and the Boltz and docking matrices do not
correlate. Two structure-based methods disagree — which **tempers** the docking-selection specificity
claim: the effect appears **method-dependent** (tied to docking into the fixed holo pocket) rather
than a robust, method-independent property of the molecules. This is consistent with the project's
recurring finding that apparent target signals are often method artifacts (the DTI-proxy affinity
artifact; the per-pocket dockability confound).

Everything ran locally on Apple-Silicon GPU (MPS): 25 Boltz-2 predictions, ~55 s each on the M5 Max.

---

## Setup

- **Method:** Boltz-2 co-folding via the `boltz-community` fork v2.8.0, `--accelerator mps` (float32;
  `aten::linalg_svd` CPU fallback). See `docs/BOLTZ_SETUP.md`. ~55 s/prediction on the M5 Max
  (1.1–2.0× a T4; far faster than the CPU-era ~1–2 h).
- **Matrix:** `M[i,j]` = Boltz-2 score of target *i*'s **top-1 docking hit** co-folded with protein
  *j*'s sequence (extracted from the holo PDB construct, longest AA chain). 5×5 = 25 runs; diagonal =
  own protein. Same 5 targets/hits as the docking POC. 3 diffusion samples for affinity; 0 NaN cells.
- **Metrics:** primary `affinity_pred_value` (log₁₀ IC₅₀ µM, **lower = stronger**); secondary
  `affinity_probability_binary` (higher = more likely binder) and `ligand_iptm` (higher = more
  confident complex — the folding-failure guard).
- **Comparison:** the same `synformer.dock.geometry.mismatch_summary` + per-pocket normalization used
  for docking, applied to the Boltz matrix; and a cell-by-cell comparison against the docking matrix
  built from the **same** top-1 hits. Reproduce with `scripts/boltz_analyze.py`.

---

## Result

### Boltz-2 affinity_pred matrix (lower = stronger)

```
 hit \ pocket    CA12     GR      KIT     RBP4   gyraseB
 CA12            0.97    0.94    0.50    0.53    0.80
 GR              0.11   -0.20   -0.55    0.34    0.66
 KIT            -0.62    0.36   -0.53    0.09    0.18
 RBP4            1.08    1.16    1.25    1.23    1.40
 gyrase-B        0.13   -0.64    0.08    0.09    0.01
```

| view | own_mean | offdiag_mean | delta | win-rate |
|---|---|---|---|---|
| Boltz raw | — | — | **−0.101** | **0.40** |
| Boltz pocket-normalized | — | — | **−0.039** | **0.60** |
| Docking (same 5 hits) raw | — | — | −2.290 | 0.80 |
| Docking (same 5 hits) normalized | — | — | **−1.743** | **1.00** |

Boltz's own-vs-mismatch delta is essentially zero (−0.04 normalized) and its win-rate is at chance
(normalized 3/5; raw 2/5) — **no target-specificity**. For the *identical* molecules, docking shows a strong, consistent
own-pocket preference (normalized win 5/5). Only GR and KIT are own-best under Boltz; CA12 and RBP4
are own-*worst*, and KIT's hit actually scores best in the CA12 pocket (−0.62), not its own (−0.53).

### Secondary metrics confirm the null

- **binder probability** — own > off-diagonal in only **1/5** targets. Several hits are flat or
  protein-driven: RBP4's hit reads as a weak binder against *every* protein (0.28–0.58); GR's hit is
  uniformly high (0.54–0.73, promiscuous) — patterns reminiscent of a molecule-driven, not
  match-driven, response.
- **ligand_iptm** — every diagonal is **0.92–0.96** and all 25 cells 0.79–0.97: the complexes folded
  **confidently everywhere**. The null is therefore *not* a folding-failure artifact — Boltz produced
  high-confidence poses and still saw no own-pocket affinity preference.

### Cross-method agreement

Boltz vs docking over the 25 shared cells: **Spearman −0.095, Pearson −0.263, sign-agreement 3/5**.
The two matrices are uncorrelated (if anything, faintly anti-correlated). The methods disagree not
just in aggregate but cell by cell.

---

## Interpretation

Docking selection reliably finds molecules that dock well into a **specific rigid crystal pocket** —
that is real and useful (Result 1 of the docking POC: selected ≥ known ligands, 5/5). But when the
same molecules are scored by an independent method that folds the complex de novo from sequence,
the **own-pocket preference disappears.** The most likely reading: the docking "specificity" is
substantially a property of shape-complementarity to the particular holo structure used, which does
not transfer to a flexibly-folded complex — so it should not be over-interpreted as intrinsic
molecular target-selectivity.

## Caveats (both directions)

- **Small, low-power:** N=5 targets, top-1 hit each, one diffusion-sample setting. This detects
  *absence of a strong, method-robust* signal; it does not prove none exists.
- **Boltz-2's own limitation cuts the other way:** Boltz-2 has a documented weakness ranking
  **generated / non-training-like** molecules (decoy-memorization; see `AFFINITY_TOOLS_RESEARCH.md`).
  The flat/promiscuous binder-probability patterns are consistent with Boltz simply not
  discriminating de-novo molecules well — so its null could partly be a *Boltz* failure, not proof the
  molecules are non-specific. **Neither method is ground truth.**
- **The two methods measure different things:** rigid-receptor docking vs de-novo co-folding; some
  disagreement is expected. The honest conclusion is that the target-specificity claim is
  **method-dependent and should be stated cautiously**, not that it is refuted.

## What this means

The value here is the corrective, cross-method discipline: a second structure-based method fails to
reproduce the docking target-specificity, so that headline should be reported as *method-dependent*
(docking-into-holo-pocket), pending stronger evidence. Docking selection remains useful for surfacing
credible binders; its *selectivity* claim needs an orthogonal, higher-power test — more targets/seeds
with confidence intervals, a co-folding tool validated on generated molecules, or ultimately
experimental binding data.

## Artifacts
- Scores: `data/boltz/boltz_scores.csv` · summary: `data/boltz/boltz_mismatch_summary.csv` (gitignored)
- Inputs: `data/boltz/matrix_inputs.json` · setup: `docs/BOLTZ_SETUP.md`
- Code: `scripts/boltz_matrix_prepare.py`, `scripts/boltz_matrix.py`, `scripts/boltz_analyze.py`
- Reproduce the analysis: `.venv/bin/python -m scripts.boltz_analyze`
