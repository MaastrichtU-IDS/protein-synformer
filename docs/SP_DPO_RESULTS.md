# SP-DPO: Per-Molecule Specificity DPO (pilot) — Results

**Date:** 2026-07-12 · Branch `powered-specificity` · Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-12-per-molecule-dpo-pilot*.md`

## The question

The one generation-side lever the project had **not** tested: can **weight updates** (not static
conditioning or frozen re-biasing) make the pocket-conditioned generator's **raw samples** target-specific?
Concretely — does DPO fine-tuning of SP-C on **per-molecule own-vs-mismatch specificity** preference pairs
make its raw samples more *own-preferring* on **held-out** pockets than the base model?

## Method (pilot)

- **Split (family-diverse):** 10 train + 4 held-out targets from the 41-target set. Train:
  KIT/JAK3/CDK5 (kinase), CA12 (lyase), GR (nuclear receptor), 5-HT1A/A1R (GPCR), RBP4 (lipocalin),
  KMO (oxidoreductase), Cathepsin G (protease). Held-out (disjoint): STK16 (kinase — the within-family
  generalization crux), 5-HT2A (GPCR), FABP4 (lipid-binding), gyraseB (bacterial ATPase — far case).
- **Pairs:** for each train target, generated 48 molecules **keeping per-molecule route tensors**, docked
  all (smina) into own + the other 9 train pockets → per-molecule specificity `z(own) − mean(z(mismatch))`
  → winner (specific) / loser (promiscuous) pairs. **624 pairs** across 10 targets (13–45 mols scored/target).
- **DPO:** policy = trainable SP-C, reference = frozen SP-C; loss `−logσ(β[(llπ_w−llref_w)−(llπ_l−llref_l)])`,
  β=0.1, lr 1e-5, 3 epochs, conditioned on each pair's pocket via `get_log_likelihood` on the routes.
- **Held-out eval (the load-bearing design choice).** *Not* `powered_analyze`'s within-pool-z + top-M —
  within-pool z forces the pool-mean delta to ≈0 by construction (false null) and top-M reintroduces
  docking-*selection* (the project's own specificity mechanism), both of which would mask the generator
  effect. Instead: sample the DPO'd model **and** base SP-C on the 4 held-out pockets, dock **both pools
  into the same shared panel** (held-out own pocket + the 10 train pockets), and compare per-molecule
  **own-preference** `d(m) = mean(mismatch_score) − own_score` (higher = more own-preferring; shared panel
  differences out per-pocket scale across pools). Origin tracked by SMILES-set membership (both pools are
  `source==candidate`). Compared mean d(DPO) vs mean d(base) with an **unpaired two-sample bootstrap**;
  cross-checked with a joint-z variant (z over base∪DPO per pocket).

## Result — NULL: in-sample preference fit, no held-out transfer

**Training fit the preference in-sample:** mean DPO margin `(llπ_w−llπ_l)−(llref_w−llref_l)` rose 2.97 → 3.49
over epochs (the policy learned to prefer the specific winners), with a moderate policy-vs-reference drift
(−0.73 → −2.09, not collapse). DPO also **shifted the generative distribution** — held-out base/DPO sample
pools overlap only ~18/48.

**But it did not transfer to held-out own-preference:**

| held-out target | family | d_base | d_dpo | Δ (DPO−base) | 95% CI (bootstrap) | joint-z Δ |
|---|---|---|---|---|---|---|
| O75716 | kinase (STK16) | −0.462 | −0.487 | −0.025 | [−0.34, +0.32] | +0.16 |
| P28223 | GPCR (5-HT2A) | +0.996 | +0.790 | −0.206 | [−0.71, +0.29] | +0.06 |
| P15090 | lipid-binding (FABP4) | −0.337 | −0.188 | +0.149 | [−0.49, +0.78] | −0.18 |
| P0C559 | bacterial ATPase (gyraseB) | −0.380 | +0.031 | +0.411 | [−0.26, +1.32] | −0.21 |
| **pooled** | | −0.046 | +0.037 | **+0.082** | **[−0.22, +0.41]** | — |

**Direction splits 2/2, every CI includes 0, the pooled difference is ns.** The raw-d and joint-z metrics
agree in direction per target (internal consistency). There is no reliable DPO>base effect on held-out
raw-sample specificity.

**ADMET unchanged** (`admet-ai`, held-out pools): base `admet_pass` 4.7% → DPO 5.6%; hERG-favorable 32.9 →
33.1%; DILI-favorable 22.8 → 26.1%; QED percentile 64.9 → 70.6; solubility p24 (flat). DPO did **not** trade
specificity for worse druggability — and the pool stays ~95% ADMET-liable (same regime as SP-AD).

**Boltz spot-check: not run** — it was pre-committed as the corroboration gate *for a win*; with no smina-level
win to corroborate, it is moot (same logic as the answered Boltz-as-selector question).

## Interpretation

DPO **can** push the generator to internalize the per-molecule specificity preference **in-sample** (margin
rises, distribution shifts) — but this **does not generalize** to making raw samples target-specific on
**unseen** pockets. This is the same wall every generator-side lever hit (SP2/SP-C/SP-L/SP-F): the model
learns *something*, but not transferable targeting. Weight updates do not escape it at this scale. The
faint positive pooled direction (+0.08, ns) and the fact that the two DPO>base targets are the two most
distinct-from-train families (FABP4, gyraseB) are **non-significant observations**, not claims.

## Decision (pre-committed rule)

The rule fixed before results: *clean DPO>base across the 4 held-out that survives a Boltz spot-check →
run the full ~7-day study; otherwise stop.* The outcome is not a clean win (2/2 split, ns), so:
**do not run the full study.** With n=4 (≈1 target/family) and 1 seed, this null is **underpowered /
inconclusive** — it is **not** a hard capstone-null, and should not be reported as "DPO cannot work." It is
evidence that a cheap pilot at these settings shows **no signal worth the full run**.

## Caveats

- **n=4 held-out, 1 seed, pilot hyperparameters** (β=0.1, lr 1e-5, 3 epochs, 624 pairs). Underpowered by
  design; a real negative would need more targets/seeds and a hyperparameter sweep.
- Specificity pairs are **smina/shape-fit** (the signal Boltz does not corroborate at the molecule level,
  per FINDINGS) — so even a positive here would have needed the Boltz gate.
- Thin train targets (O15229: 9 pairs) contributed noisy, possibly overfit gradients.
- Mismatch panel for held-out = the 10 train pockets (fixed, shared, rich); own + few-held-out-only would
  have been too scale-noisy.

## Reproduce

```
# pairs (after docking dpo_dock_driver.sh -> dpo_dock_scores.csv)
.venv/bin/python -m scripts.build_dpo_pairs --scores data/dock/dpo/dpo_dock_scores.csv \
  --targets data/dock/dpo/train10.json --out-dir data/dock/dpo/pairs
# train
.venv-train/bin/python -m scripts.dpo_train --ckpt <SP-C> --routes-dir data/dock/dpo/routes \
  --pairs-dir data/dock/dpo/pairs --out-ckpt data/ckpt/dpo_pilot.ckpt --lr 1e-5 --epochs 3 --beta 0.1
# held-out eval (after eval_dock_driver.sh -> heldout/eval_scores.csv)
.venv/bin/python -m scripts.dpo_eval_report --scores data/dock/dpo/heldout/eval_scores.csv \
  --train-json data/dock/dpo/train10.json --heldout-json data/dock/dpo/heldout4.json \
  --base-dir data/dock/dpo/heldout/base --dpo-dir data/dock/dpo/heldout/dpo \
  --out data/dock/dpo/heldout/dpo_eval_summary.json
```

Code (all TDD'd): `synformer/molopt/dpo.py` (loss + route slicing), `scripts/generate_routes.py`,
`scripts/dpo_pairs.py`, `scripts/dpo_train.py`, `scripts/dpo_eval.py`, `scripts/build_dpo_pairs.py`,
`scripts/dpo_eval_report.py`; drivers `dpo_dock_driver.sh`, `eval_dock_driver.sh`.
Artifacts under `data/dock/dpo/` and `data/ckpt/dpo_pilot.ckpt`.
