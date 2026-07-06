# SP2 Results — Richer Protein Conditioning (fair 4-way, 2026-07-06)

**Question:** does a richer protein encoder beat the study's plain `nn.Linear` encoder?

**Method (fair):** all four arms warm-start from `sf_ed_default.ckpt`, same SP2
protein-disjoint split, same budget (15k steps, lr 3e-4, batch 16), same eval (40 test
proteins, repeat 40, seed 42, `scripts/eval_variant.py`, each variant's `last.ckpt`). Only
the encoder differs. The three richer encoders use a near-zero output gate (init 0.05) so
they train (see SP2_ENCODER_COLLAPSE.md — without it they collapsed). Baseline unchanged.

All four converged comparably (val/loss_token ≈ 0.04).

| variant | validity | max-Tanimoto | uniqueness | novelty | int.div | scaffold div | SA (↓ better) | route len |
|---|---|---|---|---|---|---|---|---|
| baseline (linear) | 0.366 | 0.136 | 0.986 | 0.991 | 0.902 | 0.791 | 2.75 | 1.79 |
| transformer (self-attn+mask) | 0.720 | 0.151 | 0.993 | 0.991 | 0.883 | 0.792 | 2.31 | 1.27 |
| **masked (MLP+mask)** | **0.779** | **0.157** | 0.989 | 0.979 | 0.878 | 0.750 | **2.22** | 1.34 |
| latent (Perceiver) | 0.665 | 0.147 | 0.987 | 0.989 | 0.889 | 0.799 | 2.36 | 1.49 |

## Verdict: richer conditioning helps
- **Validity ~doubles**: 0.37 (baseline) → 0.67–0.78 (richer). Far more generations build
  into valid molecules — the single biggest gain.
- **Similarity to known ligands** improves modestly and consistently: 0.136 → 0.147–0.157
  (+8% to +15%; masked best).
- **Synthesizability** improves: SA 2.75 → 2.22–2.36 (lower = easier), routes slightly
  shorter.
- **Novelty / uniqueness / scaffold diversity** stay comparably high (~0.98 / ~0.99 / ~0.78).
  Internal diversity marginally *lower* for richer (0.88 vs 0.90) — a minor trade-off.
- **Best overall: `masked` (per-residue MLP + padding mask)** — top validity (0.78) and
  similarity (0.157) and best SA (2.22). `transformer` close behind; `latent` also beats
  baseline but by less.

## Caveats
- Similarity gains are modest and absolute similarity is still low (~0.15) — these are not
  close reproductions of known ligands, just closer than baseline.
- "Better" here = more valid / slightly more similar / more synthesizable. It is NOT a
  binding-affinity claim (SP3 showed proxy affinity scorers are protein-blind here).
- 40-protein subset, single seed, SP2 split (not the study's exact split). `masked` bundles
  MLP + masking (not isolated). Directional, not a powered benchmark.

## Takeaway
Once properly initialized, richer protein conditioning is a real improvement over the
study's linear encoder — chiefly a large validity gain plus modest similarity/
synthesizability gains — with the padding-mask + small-MLP encoder (`masked`) the best of
the three tested.
