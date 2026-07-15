# Tier-3 DAVIS Kinase Calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline) or subagent-driven. Steps use checkbox syntax.

**Goal:** Firm-or-break the Tier-2 within-kinase docking-selectivity ρ 0.245 on the dense, low-noise DAVIS panel (68 drugs × 15 kinase pockets, 78 protein-kinase pairs).

**Architecture:** Reuse the Tier-2 docking + correlation machinery. Only genuinely new code is DAVIS load/gene-mapping. Dock 68 DAVIS drugs into 15 kinase crystal pockets → per-pocket z → Spearman(measured ΔpKd, −docked Δ) per-pair + pooled, compound-clustered bootstrap.

**Tech Stack:** PyTDC (DAVIS), sklearn/scipy/rdkit (already installed), `synformer.dock` + `powered_run` (smina via `smina.static`), pandas.

## Global Constraints

- Docking venv `.venv` with `SMINA=$(pwd)/smina.static`, `CUDA_VISIBLE_DEVICES=""`; cap-4 parallel driver; never `pkill -f <scriptname>` in a launch command (self-match → SIGTERM 144).
- Metric convention = Tier-2's: docking score more-negative = better binding; z per pocket-column over the docked compound set; measured ΔpKd(A−B); ρ(measured, −docked) so **+ = docking tracks selectivity**.
- pKd = 9 − log10(Kd[nM]); DAVIS non-binders Kd 10000 → pKd 5.0.
- Primary = 13 protein kinases; robustness = all 15 (adds PIK3CD, RIOK1).
- Commit only task files via explicit `git add`; footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg`.
- Gene→pocket target_id map (verified via UniProt):
  `{KIT:P10721_WT, JAK3:P52333_WT, FGFR1:P11362_WT, CDK5:Q00535_WT, DYRK1A:Q13627_WT, CSNK1A1:P48729_WT, CSNK1G1:Q9HCP0_WT, CSNK1E:P49674_WT, PHKG1:Q16816_WT, STK16:O75716_WT, NEK1:Q96PY6_WT, CAMK4:Q16566_WT, DAPK2:Q9UIK4_WT}` (protein kinases) + robustness `{PIK3CD:O35904_WT, RIOK1:Q9BRS2_WT}`.

---

### Task 1: `davis_prep.py` — load DAVIS, map to pockets, emit dock set + measured table

**Files:** Create `scripts/davis_prep.py`, `tests/test_davis_prep.py`

**Interfaces (Produces):** `base_gene(target_id:str)->str`; `kd_to_pkd(kd_nM:float)->float`; writes `data/dock/davis/dock_set.txt` (drug SMILES/line) + `data/dock/davis/measured_davis.json` (`{smiles: {gene: pKd}}`) + `data/dock/davis/kinase_pockets.json` (`{gene: target_id}` for genes present).

- [ ] **Step 1: Failing tests**
```python
# tests/test_davis_prep.py
import math
from scripts.davis_prep import base_gene, kd_to_pkd

def test_base_gene_strips_mutation_and_domain():
    assert base_gene("ABL1(F317I)") == "ABL1"
    assert base_gene("CSNK1A1") == "CSNK1A1"
    assert base_gene("MAP3K1-domain") == "MAP3K1"
    assert base_gene("JAK3(JH1domain-catalytic)") == "JAK3"

def test_kd_to_pkd():
    assert abs(kd_to_pkd(1.0) - 9.0) < 1e-9        # 1 nM -> pKd 9
    assert abs(kd_to_pkd(10000.0) - 5.0) < 1e-9    # 10 uM (non-binder) -> pKd 5
```

- [ ] **Step 2: Run → fail** — `.venv/bin/python -m pytest tests/test_davis_prep.py -q`

- [ ] **Step 3: Implement**
```python
# scripts/davis_prep.py
"""Tier-3: load DAVIS, map kinases to our crystal pockets, emit the docking set + measured table."""
import json, math, re
from pathlib import Path

GENE_TID = {  # protein kinases (primary) + PIK3CD/RIOK1 (robustness)
    "KIT": "P10721_WT", "JAK3": "P52333_WT", "FGFR1": "P11362_WT", "CDK5": "Q00535_WT",
    "DYRK1A": "Q13627_WT", "CSNK1A1": "P48729_WT", "CSNK1G1": "Q9HCP0_WT", "CSNK1E": "P49674_WT",
    "PHKG1": "Q16816_WT", "STK16": "O75716_WT", "NEK1": "Q96PY6_WT", "CAMK4": "Q16566_WT",
    "DAPK2": "Q9UIK4_WT", "PIK3CD": "O35904_WT", "RIOK1": "Q9BRS2_WT",
}

def base_gene(target_id: str) -> str:
    return re.split(r"[(\-]", str(target_id))[0].strip().upper()

def kd_to_pkd(kd_nM: float) -> float:
    return 9.0 - math.log10(float(kd_nM))

def main():
    import os
    os.makedirs("data/tdc", exist_ok=True)
    cwd = os.getcwd(); os.chdir("data/tdc")
    from tdc.multi_pred import DTI
    df = DTI(name="DAVIS").get_data()   # Drug_ID, Drug(SMILES), Target_ID(gene), Y(Kd nM)
    os.chdir(cwd)
    df["gene"] = df.Target_ID.map(base_gene)
    df = df[df.gene.isin(GENE_TID)].copy()
    df["pkd"] = df.Y.map(kd_to_pkd)
    # median pKd per (drug SMILES, gene) across DAVIS mutant rows
    agg = df.groupby(["Drug", "gene"]).pkd.median().reset_index()
    measured = {}
    for _, r in agg.iterrows():
        measured.setdefault(r.Drug, {})[r.gene] = float(r.pkd)
    genes_present = sorted({g for m in measured.values() for g in m})
    out = Path("data/dock/davis"); out.mkdir(parents=True, exist_ok=True)
    (out / "dock_set.txt").write_text("\n".join(measured) + "\n")
    json.dump(measured, open(out / "measured_davis.json", "w"))
    json.dump({g: GENE_TID[g] for g in genes_present}, open(out / "kinase_pockets.json", "w"), indent=1)
    print(f"drugs={len(measured)} genes={len(genes_present)}: {genes_present}", flush=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → pass** (2 tests). **Step 5: Run `main` live**: `.venv/bin/python -m scripts.davis_prep` — expect ~68 drugs, 15 genes. **Step 6: Commit** (`scripts/davis_prep.py`, `tests/test_davis_prep.py`).

---

### Task 2: `davis_dock_driver.sh` — dock the DAVIS drugs into the kinase pockets (ops)

**Files:** Create `scripts/davis_dock_driver.sh`, `data/dock/davis/panelN.json`

**Interfaces (Consumes):** `dock_set.txt`, `kinase_pockets.json`. **Produces:** `data/dock/davis/dock_scores.csv` (`target,pocket,molecule,source,score`).

- [ ] **Step 1:** Build the panel JSON (the ≤15 kinase targets, subset of `powered_targets_67.json`):
```bash
.venv/bin/python -c "
import json
allt={t['target_id']:t for t in json.load(open('data/dock/powered_targets_67.json'))}
kp=json.load(open('data/dock/davis/kinase_pockets.json'))
json.dump([allt[tid] for tid in kp.values()], open('data/dock/davis/panelN.json','w'), indent=1)
print('panel:', list(kp.values()))
"
```

- [ ] **Step 2:** Write `scripts/davis_dock_driver.sh` — shard the dock_set 4 ways, each shard docked (as one source's candidates) into all panel pockets, cap-4 parallel. Model on `tier2_dock_driver.sh` (same structure): put each shard's SMILES as `data/dock/davis/shard{i}/cand/<first-panel-tid>.txt`, run `powered_run --targets data/dock/davis/panelN.json --candidates-dir data/dock/davis/shard{i}/cand --sources <first-panel-tid> --scores work_davis/shard{i}/s.csv --n-candidates 200 --top-m 200 --n-refs 0 --skip-af --work-dir work_davis/shard{i}/wd`, then merge shards (dedup molecule,pocket) → `data/dock/davis/dock_scores.csv`. Reuse the exact env header (`SMINA`, `CUDA_VISIBLE_DEVICES`) and `throttle()` from `tier2_dock_driver.sh`.

- [ ] **Step 3:** Launch detached (`nohup bash scripts/davis_dock_driver.sh > logs/davis/driver.log 2>&1 &`, no self-matching pkill), watch by coverage for "DAVIS DOCK ALL DONE" (~1h, ~1020 docks). Health-check first rows.

- [ ] **Step 4: Commit** the driver + panelN.json (panelN.json is under `data/` symlink — commit driver only if data path rejects).

---

### Task 3: `davis_analyze.py` — measured-vs-docked selectivity (reuse Tier-2 metric)

**Files:** Create `scripts/davis_analyze.py`, `tests/test_davis_analyze.py`

**Interfaces (Consumes):** `dock_scores.csv`, `measured_davis.json`, `kinase_pockets.json`, `scripts.dpo_eval.two_sample_diff_ci`, `scipy.stats.spearmanr`.

- [ ] **Step 1: Failing test** for the one new pure helper — per-pair ρ distribution aggregation:
```python
# tests/test_davis_analyze.py
from scripts.davis_analyze import summarize_pairs

def test_summarize_pairs_counts_positive():
    # three pairs with rhos +0.3, +0.1, -0.2 -> 2/3 positive, median +0.1
    per_pair = {("A","B"): 0.3, ("A","C"): 0.1, ("B","C"): -0.2}
    s = summarize_pairs(per_pair)
    assert s["n_pairs"] == 3 and s["n_positive"] == 2
    assert abs(s["median_rho"] - 0.1) < 1e-9
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** `davis_analyze.py`: load scores + measured; per-pocket z over the dock_set (best/min score per molecule-pocket, `(x-mean)/std`); for each gene pair (A,B) present and each drug measured on both, `measured Δ = pKd(A)-pKd(B)`, `docked Δ = z(tid_A)-z(tid_B)`; per-pair `spearmanr(measured, -docked)`; `summarize_pairs(per_pair_rho)->{n_pairs,n_positive,median_rho}`; pooled ρ + `two_sample_diff_ci`-style compound-clustered bootstrap (resample drugs, recompute pooled ρ, 2.5/97.5 pct); print PRIMARY (13 protein kinases) and ROBUSTNESS (all 15); write `data/dock/davis/davis_summary.json`. Import pandas/numpy/scipy at top.

```python
def summarize_pairs(per_pair_rho: dict) -> dict:
    import numpy as np
    vals = [v for v in per_pair_rho.values() if v == v]
    return {"n_pairs": len(vals), "n_positive": int(sum(v > 0 for v in vals)),
            "median_rho": float(np.median(vals)) if vals else float("nan")}
```

- [ ] **Step 4: Run → pass. Step 5: Commit** (`scripts/davis_analyze.py`, `tests/test_davis_analyze.py`).

---

### Task 4: Run, write results, update FINDINGS/CAPSTONE, finish

- [ ] **Step 1:** After docking completes, run `.venv/bin/python -m scripts.davis_analyze`; capture PRIMARY (13-kinase, 78-pair) pooled ρ + CI + per-pair distribution, ROBUSTNESS (15).
- [ ] **Step 2:** Write `docs/TIER3_DAVIS_RESULTS.md` — firm-or-break verdict vs Tier-2 0.245, per-pair distribution, honest magnitude, caveats (docking heterogeneous crystal ligands; DAVIS single-assay low-noise; PIK3CD/RIOK1 class).
- [ ] **Step 3:** Advisor-review the verdict; update `FINDINGS.md` §E (the Tier-2 row → firmed/broken) and `CAPSTONE.md` (the ρ 0.245 verdict) accordingly.
- [ ] **Step 4:** Commit docs; push `origin/powered-specificity`; update `.superpowers/sdd/progress.md`.

---

## Self-review notes
- **Spec coverage:** Task 1 = §2 data/mapping; Task 2 = §3.1 docking; Task 3 = §3.2–4 metric/decision; Task 4 = §5 results + §1 firm-or-break. ✓
- **New code is minimal** (base_gene, kd_to_pkd, summarize_pairs — all TDD'd); the z + Spearman + bootstrap reuse Tier-2's verified convention.
- **Reuse risk:** ensure `davis_analyze` z/sign convention matches `tier2_analyze` exactly (more-negative docked = better; ρ(measured, −docked)).
- **Ops:** no self-matching pkill; monitor by coverage; `data/` is a symlink (panelN.json/dock_scores under it won't git-track — expected, they're data).
