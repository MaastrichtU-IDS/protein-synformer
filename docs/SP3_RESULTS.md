# SP3 Results — Affinity Validation

## 1. Negative controls (2026-07-04) — VERDICT: the affinity result is a scorer artifact

`scripts/affinity_controls.py`, DeepPurpose `MPNN_CNN_DAVIS`, 194 test proteins, r=44.

| condition | mean_cond_best (pKd) | pct_beats_best |
|---|---|---|
| real (model generations) | 12.78 | 70.6% |
| A_mismatch (real mols, permuted protein) | 12.78 | 70.6% |
| B_foreign (other targets' ligands) | 12.83 | 71.6% |
| D_random_real (random Enamine blocks) | 13.80 | 74.7% |
| C_notrain (n=6, unreliable) | 5.98 | 16.7% |
| (reference) mean_true_best | 10.92 | — |

**Interpretation:** scoring the *same generated molecules against a deliberately
mismatched (permuted) protein* yields an identical 70.6% — the scorer is effectively
insensitive to protein identity. Random building blocks beat the native best ligand MORE
often than the model's molecules (74.7% vs 70.6%), and real validated ligands score LOWER
(10.92) than random fragments (13.80). The SP1/paper "generated molecules bind better than
known ligands" result is therefore **not target-specific and not real** — it is dominated
by molecule-level features the DAVIS-trained DTI model rewards regardless of the target.

**Consequences:**
- The paper's abstract claim of "higher-affinity candidates" is not supported by this
  proxy; report as a negative/corrective finding.
- Affinity-guided sampling *by this scorer* (SP3 Task 4) is meaningless — it would select
  for scorer-preferred molecular features, not binding. Defer until a target-sensitive
  signal is found.
- Priority shifts to: does ANY proxy show target-specificity? -> second DTI scorer
  (Task 2) and, decisively, docking (Task 3).

## 2. Second scorer (2026-07-04) — artifact is GENERAL across DTI proxies

`scripts/affinity_agreement.py`, 194 proteins, seed 42.

| scorer | pct_beats_real | pct_beats_mismatch |
|---|---|---|
| MPNN_CNN_DAVIS | 70.6% | 70.6% (identical) |
| MPNN_CNN_BindingDB | 67.0% | 64.9% (~2 pp) |

Per-protein Spearman(best-generated affinity) between the two scorers: **0.345** (weak).

**Interpretation:** BindingDB is also ~protein-blind (real vs mismatch differ by ~2 pp,
within noise). Both proxies say the molecules beat the native ligand ~65-70% of the time
regardless of which protein they're scored against, and the two scorers barely correlate
with each other. Sequence-based DTI proxies (DeepPurpose) cannot measure target-specific
affinity for this task. The corrective finding from Section 1 is robust.

## 3. Docking cross-check — the only remaining target-specific route (structure-based); optional/heavy
## 4. Guided sampling — dropped (no target-sensitive proxy to guide by)

## Overall verdict (affinity axis)
The study's "higher-affinity candidates" claim is **not supported** by available proxy
scorers, and this is a robust, corrective methodological finding. "Effectiveness" should
rest on the defensible metrics — max-Tanimoto similarity to known ligands (Table III),
synthesizability (SA ~2.4), novelty/diversity — plus the SP2 encoder-architecture
comparison. Structure-based docking is the only path to a target-specific affinity signal
and is optional given its setup cost.
