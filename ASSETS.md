# Assets catalogue

Status of every data/weight artifact prot2drug needs. `data/**` is gitignored;
this file is the source of truth. The heavy artifacts live in a single store at
`~/code/prot2drug/data/` (12 GB) and are **symlinked** into the paths the code
expects under `protein-synformer/data/` (no duplication).

## Present (provided by the team, wired via symlinks 2026-07-04)
| Expected path (under protein-synformer/data/) | Source file in ~/code/prot2drug/data/ |
|---|---|
| trained_weights/epoch=23-step=28076.ckpt (861M) | trained_weights/4-6-2025/epoch=23-step=28076.ckpt |
| trained_weights/sf_ed_default.ckpt (2.6G) | trained_weights/sf_ed_default-003.ckpt |
| trained_weights/big_pretrained_last4.ckpt (1.6G) | "trained_weights/_Big_ + Pretrained Weights + Last 4/last.ckpt" |
| protein_embeddings/embeddings_selection_float16_4973.pth (6.1G) | embeddings_selection_float16_4973-001.pth |
| synthetic_pathways/filtered_pathways_370000.pth | synthetic_pathways/filtered_pathways_370000.pth |
| building_blocks/Enamine_..._253345cmpd_20250212.sdf (428M) | same (2025 catalogue; see note) |
| evaluations/epoch=23-step=28076/infos_2025-06-11_09-12-36.pkl | infos_2025-06-11_09-12-36.pkl (saved generations, 300 proteins) |

Also in the store (not yet wired): papyrus full sets + selections
(`papyrus_selection_{123236,182129}.csv`), extra pathways (105000/290000/5),
256-bit index (`pkl-files-256bit/`), PDB files (`put in data-/`),
`synformer_ligands_test_v2025-04-02.csv` (SynFormer projection output → Fig-4
coverage data), `config/wandb.yml`.

## In-repo (committed) fixtures
- data/rxn_templates/comprehensive.txt
- data/enamine_smiles_1k.txt, data/chembl_filtered_1k.txt (smoke)
- data/*_mini.* toy embeddings/pairs/pathways
- data/other/aa_seq_test.csv — cached AA sequences for test proteins (affinity eval)

## GAPS / caveats
- **comp_2048 index MISSING.** The trained model (`hparams.yaml`) uses 2048-bit
  Morgan and expects `data/processed/comp_2048/{fpindex,matrix}.pkl`. The provided
  256-bit index is dimensionally incompatible (head predicts a 2048-bit fp).
  → regenerate from the Enamine SDF (`scripts/preprocess.py` with morgan_n_bits=2048)
  or fetch from HF `whgao/synformer`. Needed only to **re-sample**; NOT needed to
  recompute similarity from the saved `infos` pickle.
- **Exact split files** (`papyrus_{train_155187,val_19399,test_19399}.csv`) not
  provided by those names. `papyrus_selection_182129.csv` covers 294/300 eval
  targets and was used as ground truth for the Table III reproduction below.
- **Enamine version drift:** provided catalogue is 253,345 cmpd (2025-02-12); the
  study used 223,244 cmpd (2023-10-01). Regenerated indices will differ slightly.

## Reproduction status
- **Table III (similarity) — REPRODUCED (2026-07-04)** from the saved `infos` +
  `papyrus_selection_182129.csv`, no retraining/GPU:
  best-per-(protein,molecule): mean 0.1797 / median 0.1724 (report `last1`: 0.1832 / 0.1733).
- **Evaluation harness — DONE** (`synformer/eval/`, `scripts/run_eval.py`). On the saved
  300-protein eval: validity 0.442, uniqueness 0.736, novelty(vs known ligands) 0.949,
  per-protein internal diversity 0.809, scaffold diversity 0.471, mean SA 2.42 (easy to
  synthesize), route length mean 1.71 / max 5.
- **End-to-end generate pipeline — WORKS.** The `big_pretrained_last4.ckpt` model
  (last-4-layers fine-tune) re-samples 32/32 valid for P56528_WT using our rebuilt
  comp_2048 index. So re-sampling and the 2025-catalogue index are both fine.
- **LoRA-failure finding — INDEPENDENTLY REPRODUCED.** `epoch=23-step=28076.ckpt` has
  `lora=true, rank=64`; it re-samples only ~1/32 valid (trivial fragments). This matches
  the report's conclusion that LoRA essentially failed to learn. (Earlier "catalogue
  drift" hypothesis was FALSIFIED: the HF `whgao/synformer` 2048-bit index gives the same
  1/32 for the LoRA model, and the last-4 model gives 32/32 with our index.)
  → The saved `infos` (95/100 valid, sim ~0.18) came from a GOOD model (last-N), not the
  LoRA checkpoint; the `evaluations/epoch=23...` folder name is misleading.
- **Binding affinity (RQ2, the paper's biggest gap) — MEASURED (2026-07-04)** via
  DeepPurpose `MPNN_CNN_DAVIS` over 194 proteins (`scripts/eval_affinity.py`):
  best generated vs best known ligand (pKd): 12.78 vs 10.92; 70.6% of proteins have a
  generated molecule >= best known ligand; 12% of all generations beat the best known
  ligand. Proxy scorer, not experimental — directional evidence for the abstract's claim.
- **Fig-4 REAL-space coverage — REPRODUCED (2026-07-04)** via `scripts/reproduce_coverage.py`
  on `synformer_ligands_test_v2025-04-02.csv` (3090 ligands, ~1.5 tries each):
  19.2% exact REAL-space matches (report: ~20% at 8 tries, ~22% at 16).
- **notrain baseline (Table III col 1)** — `scripts/sample_notrain.py` builds the baseline
  (pretrained decoder+heads, reinit cross-attn, untrained protein encoder) and samples;
  running on a 60-protein subset to estimate the ~0.154 baseline similarity.
- **notrain baseline (Table III col 1) — REPRODUCED** on 60-protein MPS subset:
  best-per-pair mean 0.132 (report 0.154), below the fine-tuned 0.180 — qualitative
  Table III result (fine-tune > baseline) holds.
- **Local fine-tuning is VIABLE (benchmarked `scripts/bench_train.py`, MPS).** Big model
  (178M): full-FT 0.56 s/step (len 512) to 0.74 s (len 2010); last-4 0.44-0.68 s; peak
  memory <=11 GB (128 GB available). A 28k-step run = ~3.4-5.8 h. GPU sampling/retrieval
  also MPS-enabled (matmul fingerprint retrieval). No external GPU needed for SP2.
- Redundant: `data/processed/comp_hf/` (removed) — our rebuilt comp_2048 works.
- Pending: exact test split for exact-match numbers; loss curves (Figs 5-7, need
  TensorBoard logs or retrain); model-size Small-vs-Big comparison.
