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

## 2. Second scorer — pending
## 3. Docking cross-check — pending (now higher priority)
## 4. Guided sampling — deferred pending a target-sensitive scorer
