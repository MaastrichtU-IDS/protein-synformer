# Affinity-evaluation tools for a no-CUDA Mac — literature scan (2026-07-04)

Deep-research (23 sources, 24/25 verified claims). Motivated by SP3's finding that
sequence-based DTI proxies are protein-blind. Question: what modern tool can validate
*target-specific* binding of *generated* molecules, runnable on Apple Silicon (no CUDA)?

## Tool landscape (no-CUDA Mac)
- **AutoDock Vina** — Apache-2.0, native ARM/CPU, no GPU needed (CLI binary runs natively;
  pip bindings need a source build). Physics-based docking. [vina docs; ccsb-scripps/AutoDock-Vina]
- **Boltz-2** — MIT, open weights, jointly predicts 3D complex **and** a structure-based,
  target-specific affinity (`affinity_probability_binary` = binder-vs-decoy;
  `affinity_pred_value` = log10 IC50 µM). **MPS support added via PR #527** (CPU otherwise,
  slow). The one co-folding model with an affinity head categorically different from the
  DTI proxies that failed us. [jwohlwend/boltz; jclinic.mit.edu]
- **gnina** — CNN rescoring, but **requires CUDA ≥12 to build**, no Apple-Silicon/MPS path
  → poor Mac fit. [gnina/gnina]
- **AlphaFold3** — weights license-restricted (not freely redistributable) → fails the
  open-source priority. DiffDock/gnina/most ML dockers are CUDA-oriented.

## The critical, sobering evidence (why this is hard)
- **Co-folding pose confidence does NOT screen.** AF3 ligand-pLDDT separates binders from
  decoys worse than docking, sometimes worse than random (D4 AUC 46.4). [biorxiv 696505]
- **Co-folding models pattern-match, not physics.** >half of AF3-correct poses are retained
  even after the binding site is ablated — a direct structural analogue of our own
  DTI-blindness finding. [nature s41467-025-63947-5]
- **Boltz-2 has the same artifact.** Independent tests found "profound memorization
  artifacts" — retained screening enrichment even with binding sites ablated — and only
  weak-moderate correlation on diverse/novel targets (r=0.24–0.45); the "FEP-comparable"
  claim was **refuted** (0-3). [arxiv 2603.05532]
- **Prospective study (Fraser/Shoichet, 557 Mac1 complexes + 3 screens):** co-folding was
  **not** consistently better than classical docking, often worse. [biorxiv 696505]
- **Decoy bias is real:** apparent enrichment degrades sharply as actives become
  dissimilar to training ligands. Even best docking on unbiased LIT-PCBA is ~random. [RSC d5sc06481c]
- **One positive:** on MF-PCBA, Boltz-2 (mean AP 0.084) far outscored Vina (0.012) and
  gnina (0.016) at binder-vs-decoy — so it *can* help, benchmark-dependent. [arxiv 2508.17555]

## Bottom line
There is **no tool that reliably validates target-specific affinity for de-novo molecules
on a no-CUDA Mac** — the field itself is skeptical. Best available: **Boltz-2** (structure-
based, target-specific in principle, MPS-capable, explicit binder-vs-decoy head) — strictly
better than the classic Vina/smina pipeline for our purpose. BUT it may carry the same
site-insensitivity artifact we found, so **our own controls (mismatched protein / binding-
site ablation + known-actives-vs-matched-decoys) must be run on it too**. Treat any score
as a hypothesis; physics (MD/FEP) is the only rigorous validation and is out of scope here.

**Implication for the paper:** our negative-control methodology is exactly what the 2025–26
literature says is required and is missing from most generative-model papers — the
corrective finding is a genuine contribution regardless of which tool we pick.
