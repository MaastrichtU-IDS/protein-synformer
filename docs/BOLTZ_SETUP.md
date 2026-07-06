# Boltz-2 MPS Setup (boltz-community fork)

## Overview

This document records the verified Boltz-2 setup on Apple Silicon (MPS) for the boltz-validation sub-project.
Run on M5 Max, macOS, arm64.

---

## Fork and Version

| Item | Value |
|---|---|
| Fork | [Novel-Therapeutics/boltz-community](https://github.com/Novel-Therapeutics/boltz-community) |
| Version / tag | `v2.8.0` (commit `a16ba22c40085ef959a6b6d2f5d40c611dc1fffa`) |
| PyPI package name | `boltz-community` |
| Torch version (pulled in) | `2.12.1` |
| Python | `3.12` |

**Note:** The package name is `boltz-community` (not `boltz`). Install without the `boltz ==` name-match constraint.

---

## Install

```bash
cd /Users/micheldumontier/code/prot2drug/protein-synformer

# Create fresh venv (do NOT touch .venv-boltz)
python3.12 -m venv .venv-boltz-mps

# Install
.venv-boltz-mps/bin/pip install --upgrade pip
.venv-boltz-mps/bin/pip install "git+https://github.com/Novel-Therapeutics/boltz-community@v2.8.0"
```

Verification:
```bash
.venv-boltz-mps/bin/boltz predict --help 2>&1 | grep -i accelerator
# Expected: --accelerator [gpu|cpu|tpu|mps]

.venv-boltz-mps/bin/python -c "import torch; print('mps', torch.backends.mps.is_available())"
# Expected: mps True
```

---

## MPS Run Command Template

```bash
cd /Users/micheldumontier/code/prot2drug/protein-synformer
mkdir -p boltz_in boltz_out/<run_name>
time .venv-boltz-mps/bin/boltz predict boltz_in/<input>.yaml \
    --use_msa_server \
    --accelerator mps \
    --out_dir boltz_out/<run_name> \
    --output_format pdb \
    --diffusion_samples_affinity 3
```

**`--diffusion_samples_affinity`:** Set to **3** for speed (one structure + 3 affinity diffusion samples).
Increase to 5–10 for more stable affinity estimates in production runs.

---

## Smoke Run Results (P02753_WT / RBP4, 175 aa)

Input file: `boltz_in/smoke_mps.yaml`
- Protein: RBP4 P02753_WT (175 aa)
- Ligand SMILES: `O=C(CC(=O)c1ccc2ccccc2c1Br)c1ccccc1`

Command run:
```bash
time .venv-boltz-mps/bin/boltz predict boltz_in/smoke_mps.yaml \
    --use_msa_server --accelerator mps --out_dir boltz_out/smoke_mps \
    --output_format pdb --diffusion_samples_affinity 3
```

**Wall-clock time: ~121 seconds (≈2 minutes)** on M5 Max.

Confirmed log line: `GPU available: True (mps), used: True`
Failed examples: 0

---

## Output JSON Path Globs

Given an input file named `<stem>.yaml` and `--out_dir <out_dir>`:

| File | Glob |
|---|---|
| Affinity JSON | `<out_dir>/boltz_results_<stem>/predictions/<stem>/affinity_<stem>.json` |
| Confidence JSON | `<out_dir>/boltz_results_<stem>/predictions/<stem>/confidence_<stem>_model_0.json` |

For this smoke run (`out_dir=boltz_out/smoke_mps`, `stem=smoke_mps`):
- `boltz_out/smoke_mps/boltz_results_smoke_mps/predictions/smoke_mps/affinity_smoke_mps.json`
- `boltz_out/smoke_mps/boltz_results_smoke_mps/predictions/smoke_mps/confidence_smoke_mps_model_0.json`

---

## Affinity JSON Schema

```json
{
    "affinity_pred_value": 1.2127,
    "affinity_probability_binary": 0.4339,
    "affinity_pred_value1": 1.8390,
    "affinity_probability_binary1": 0.3860,
    "affinity_pred_value2": 0.5863,
    "affinity_probability_binary2": 0.4818
}
```

Keys used by downstream tasks:
- **`affinity_pred_value`**: primary affinity prediction (ensemble mean of the 3 diffusion samples).
  Units: **log₁₀(IC₅₀ / µM)** — i.e. **lower = stronger binder**.
  Example: `1.21` ≈ IC₅₀ ~16 µM (weak); a strong binder would be negative (e.g. `-1` ≈ IC₅₀ ~0.1 µM).
- **`affinity_probability_binary`**: probability of being a "binder" by a binary classifier threshold (0–1); higher = more likely a binder.
- `affinity_pred_value1`, `affinity_pred_value2`: individual diffusion samples 1 and 2.

**Sign convention:** LOWER `affinity_pred_value` = STRONGER binder.

---

## Confidence JSON Schema and `ligand_iptm` Key

```json
{
    "confidence_score": 0.9108,
    "ptm": 0.9678,
    "iptm": 0.9315,
    "ligand_iptm": 0.9315,
    "protein_iptm": 0.0,
    ...
}
```

The `ligand_iptm` key is a **top-level key** in the confidence JSON (not nested).
It is numerically equal to `iptm` for protein–small-molecule complexes (since the only cross-chain pair is protein↔ligand).

Key used downstream: **`ligand_iptm`** (top-level key in `confidence_<stem>_model_0.json`).

For this smoke run (RBP4 + bromonaphthalene ketone ligand):
- `ligand_iptm`: 0.9315 (high confidence — the ligand is well-placed relative to the protein)
- `affinity_pred_value`: 1.2127 (≈ IC₅₀ ~16 µM; weak/moderate binder as expected for this exploratory compound)
- `affinity_probability_binary`: 0.434 (borderline binary prediction)

---

## Known Warnings / Notes

### `aten::linalg_svd` CPU fallback

During every MPS run, the following warning appears — this is expected and benign:

```
UserWarning: The operator 'aten::linalg_svd' is not currently supported on the MPS backend
and will fall back to run on the CPU.
  U, S, V = torch.linalg.svd(
```

Source: `boltz/model/loss/diffusionv2.py:52`. The SVD runs on CPU while the rest of the diffusion runs on MPS. No action needed; this is a known limitation of the MPS backend for linalg operations.

---

## Existing `.venv-boltz` (do NOT modify)

The repo also contains `.venv-boltz` which uses a different (older) Boltz version that does NOT expose `--accelerator mps`. Leave it in place; it is used by the non-MPS baseline runs.

---

## Boltz Model Weights Cache

Weights are downloaded automatically on first run to the default cache location:
`~/.boltz/` (or `~/.cache/boltz/` depending on the version).

With a warm pip cache and pre-downloaded weights, the first run is fast.
