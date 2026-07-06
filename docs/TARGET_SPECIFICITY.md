# Is the protein-conditioned SynFormer target-specific? (2026-07-06)

Three independent controls test whether the model does anything *specific to the target
protein* (vs generating generically drug-like, synthesizable molecules). All apply the same
logic: compare the true (protein, ligand) association against a mismatched/shuffled one.

## 1. Ligand-similarity control (the SP1 / Table III metric)
Max-Tanimoto of generated molecules to known ligands, own vs mismatched protein (294 proteins):
- gen(P) vs **P's own** ligands: **0.180**
- gen(P) vs **mismatched** proteins' ligands: **0.202**
→ generated molecules are NOT more similar to the target's own ligands than to random
proteins' ligands. Table III's ~0.18 reflects generic drug-likeness overlap, not targeting.

## 2. Proxy-affinity control (SP3, DeepPurpose MPNN_CNN)
- real vs mismatched-protein: identical (70.6% == 70.6%); random Enamine blocks beat known
  ligands 74.7%; second scorer (BindingDB) same pattern.
→ the DTI proxies are protein-blind; no target-specific affinity signal (see SP3_RESULTS.md).

## 3. Model-likelihood control (this analysis)
Fix the molecule, vary the protein: LL(pathway(L) | P) matched vs mismatched. Molecule-
intrinsic likelihood cancels, so any difference is pure protein conditioning.
- **Paired** (n=150, 1 shuffle): study_last4 win-rate 53.3%; sp2_masked 55.3% (both ~chance).
- **Retrieval** (n=60 ligands, K=10 decoy proteins, n=600 comparisons):
  | model | top-1 (rand 9.1%) | mean rank /11 (rand 6.0) | pairwise win (rand 50%) |
  |---|---|---|---|
  | study_last4 | 13.3% | 4.87 | 61.3% |
  | sp2_masked  | 6.7%  | 5.85 | 51.5% |
→ the study's model shows a **weak but consistent** above-chance signal (it ranks the true
protein better than random by its own likelihood); the masked model shows ~none here. The
paired vs retrieval tests disagree on which model is better → that model-level difference is
within noise. Neither is strongly target-specific.

## Verdict
The protein-conditioned SynFormer learned only **faint** target-specificity — detectable in
the study model's likelihood (weakly), but NOT in generated-molecule similarity or proxy
affinity. It produces valid, novel, synthesizable molecules but is **not a demonstrably
target-specific generator**. SP2's richer conditioning improves generic quality
(validity ~2x, SA) but does **not** measurably improve targeting.

## Caveats
Small N (60-150), single seeds, crude proxies (Tanimoto to a sparse known-ligand set;
DeepPurpose). This shows we cannot *detect* strong targeting, not that none exists — a real
binder via a novel scaffold would evade Tanimoto, and the likelihood test is a coarse probe.
A fully powered version (larger N, multiple seeds, CIs; structure-based docking e.g. Boltz-2)
would firm the numbers, but the qualitative conclusion is stable across all three controls.
