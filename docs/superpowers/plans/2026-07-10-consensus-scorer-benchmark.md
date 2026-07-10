# SP-CS Consensus-Scorer Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a consensus of smina + Boltz discriminates known binders from random decoys more robustly (worst-case AUROC) than either scorer alone, across 4 targets.

**Architecture:** Pure analysis. Reuse existing smina known/random (`dock_scores.csv`) and generate Boltz known/random with `boltz_controls`; a new `consensus_score.py` joins them, computes rank-mean and z-sum consensus, and reports per-target AUROC with mean + worst-case.

**Tech Stack:** Python 3.10, pandas, numpy, scikit-learn (`roc_auc_score`, confirmed present), click, pytest; Boltz-2 via `.venv-boltz`. Spec: `docs/superpowers/specs/2026-07-10-consensus-scorer-benchmark-design.md`.

## Global Constraints

- **Analysis + tests run in `.venv`** (`.venv/bin/python -m pytest`). Boltz generation runs in `.venv-boltz`.
- **Boltz needs the proxy:** its `--use_msa_server` reaches `api.colabfold.com`; set `https_proxy=http://proxy.unimaas.nl:3128/` (and HTTP_PROXY etc.) or all MSA requests fail. Also `BOLTZ=.venv-boltz/bin/boltz` (default points at a nonexistent `.venv-boltz-mps`), `CUDA_VISIBLE_DEVICES=0`, `--no-kernels`.
- **4 usable targets:** `O43570_WT, P10721_WT, P02753_WT, P0C559_WT` (P06537 dropped: only 3 knowns). Require ≥5 knowns per target; skip + log otherwise.
- **Scorer direction:** lower is stronger for both (smina kcal/mol, Boltz `affinity_pred`); "strength" = `−score`.
- **Consensus within each target** over its known∪random molecules; benchmark on the smina∩Boltz intersection.
- **Win = consensus worst-case (min) AUROC > each single scorer's worst-case**; also report mean.
- **Commits:** author `michel.dumontier@gmail.com`; footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session:` line. Commit only task files via explicit `git add <path>` — never `git add -A` (working tree has pre-existing drift).
- **smina known/random already exist** in `data/dock/dock_scores.csv` (own-pocket rows, `source ∈ {known,random}`); no new docking.

---

### Task 1: `consensus_score.py` — join, consensus, per-target AUROC

**Files:**
- Create: `scripts/consensus_score.py`
- Test: `tests/test_consensus_score.py`

**Interfaces:**
- Consumes: `pandas`, `numpy`, `sklearn.metrics.roc_auc_score`.
- Produces:
  - `load_smina(dock_scores_csv, targets) -> pd.DataFrame` cols `[target, molecule, is_known, smina]` — own-pocket (`pocket==target`) rows with `source ∈ {known,random}`, restricted to `targets`; `is_known = source=='known'`; `smina = score`.
  - `load_boltz(boltz_csv) -> pd.DataFrame` cols `[target, molecule, boltz]` — from `boltz_controls` output (`smiles→molecule`, `affinity_pred→boltz`).
  - `build_frame(smina_df, boltz_df) -> pd.DataFrame` — inner-join on `(target, molecule)`, cols `[target, molecule, is_known, smina, boltz]`.
  - `benchmark(frame, min_known=5) -> dict` — per target with ≥`min_known` knowns and ≥2 of each class: AUROC (y=`is_known`, score=strength) for `smina`, `boltz`, `rankmean`, `zsum`; plus `{"per_target": {...}, "mean": {scorer: v}, "worst": {scorer: v}, "skipped": [targets]}`. `strength = −score`; `rankmean` = mean of within-target ranks of `smina` and `boltz` strengths; `zsum` = sum of within-target z-scored strengths.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_consensus_score.py
import numpy as np
import pandas as pd
from scripts.consensus_score import build_frame, benchmark


def _frame():
    # Target GOOD: both scorers rank knowns above randoms (AUROC 1.0 each).
    # Target SMINA_FAILS: smina INVERTED (knowns look weak by smina) but Boltz correct;
    #   consensus should beat smina here (worst-case rescue).
    rows = []
    # GOOD target: 3 known (strong = very negative), 3 random (weak)
    for i, s in enumerate([-9, -8, -7]):
        rows.append(("GOOD", f"k{i}", True, s, s))       # smina & boltz agree strong
    for i, s in enumerate([-4, -3, -2]):
        rows.append(("GOOD", f"r{i}", False, s, s))
    # SMINA_FAILS target: knowns are Boltz-strong but smina-weak; randoms the opposite
    for i in range(3):
        rows.append(("SMINA_FAILS", f"k{i}", True, -2.0 - i * 0.1, -9.0 + i * 0.1))  # smina weak, boltz strong
    for i in range(3):
        rows.append(("SMINA_FAILS", f"r{i}", False, -9.0 + i * 0.1, -2.0 - i * 0.1))  # smina strong, boltz weak
    return pd.DataFrame(rows, columns=["target", "molecule", "is_known", "smina", "boltz"])


def test_benchmark_auroc_and_worstcase_rescue():
    out = benchmark(_frame(), min_known=3)
    # GOOD: everyone perfect
    assert out["per_target"]["GOOD"]["smina"] == 1.0
    assert out["per_target"]["GOOD"]["boltz"] == 1.0
    # SMINA_FAILS: smina AUROC ~0 (inverted), boltz ~1, consensus in between but > smina
    sf = out["per_target"]["SMINA_FAILS"]
    assert sf["smina"] < 0.5 and sf["boltz"] > 0.9
    assert sf["rankmean"] > sf["smina"]
    # worst-case (min across targets): consensus rescues smina's catastrophic target
    assert out["worst"]["rankmean"] > out["worst"]["smina"]


def test_build_frame_inner_join_drops_unmatched():
    smina = pd.DataFrame({"target": ["T", "T"], "molecule": ["a", "b"],
                          "is_known": [True, False], "smina": [-8.0, -3.0]})
    boltz = pd.DataFrame({"target": ["T", "T"], "molecule": ["a", "c"], "boltz": [-7.0, -1.0]})
    f = build_frame(smina, boltz)
    assert list(f["molecule"]) == ["a"]   # only 'a' in both


def test_benchmark_skips_low_known_target():
    df = pd.DataFrame([("T", "k0", True, -9, -9), ("T", "r0", False, -2, -2),
                       ("T", "r1", False, -1, -1)],
                      columns=["target", "molecule", "is_known", "smina", "boltz"])
    out = benchmark(df, min_known=5)
    assert "T" in out["skipped"] and "T" not in out["per_target"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_consensus_score.py -q`
Expected: FAIL (`ModuleNotFoundError: scripts.consensus_score`).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/consensus_score.py
"""Consensus-scorer discrimination benchmark: does consensus(smina, Boltz) separate
known binders from random decoys more robustly (worst-case AUROC) than either alone?"""
from __future__ import annotations

import click
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_smina(dock_scores_csv, targets) -> pd.DataFrame:
    d = pd.read_csv(dock_scores_csv)
    d = d[(d.target.isin(targets)) & (d.pocket == d.target) & (d.source.isin(["known", "random"]))]
    out = d[["target", "molecule", "source", "score"]].copy()
    out["is_known"] = out.source == "known"
    out = out.rename(columns={"score": "smina"}).drop(columns=["source"])
    return out.dropna(subset=["smina"]).drop_duplicates(["target", "molecule"])


def load_boltz(boltz_csv) -> pd.DataFrame:
    d = pd.read_csv(boltz_csv).rename(columns={"smiles": "molecule", "affinity_pred": "boltz"})
    return d[["target", "molecule", "boltz"]].dropna(subset=["boltz"]).drop_duplicates(["target", "molecule"])


def build_frame(smina_df, boltz_df) -> pd.DataFrame:
    return smina_df.merge(boltz_df, on=["target", "molecule"], how="inner")


def _auroc(y_known, strength) -> float:
    return float(roc_auc_score(y_known.astype(int), strength))


def benchmark(frame, min_known: int = 5) -> dict:
    per_target, skipped = {}, []
    for target, g in frame.groupby("target"):
        n_known = int(g.is_known.sum())
        n_rand = int((~g.is_known).sum())
        if n_known < min_known or n_rand < 2:
            skipped.append(str(target))
            continue
        s_smina = -g.smina.to_numpy(dtype=float)   # strength = -score
        s_boltz = -g.boltz.to_numpy(dtype=float)
        rankmean = (pd.Series(s_smina).rank().to_numpy() + pd.Series(s_boltz).rank().to_numpy()) / 2.0
        zsum = _z(s_smina) + _z(s_boltz)
        y = g.is_known
        per_target[str(target)] = {
            "smina": _auroc(y, s_smina), "boltz": _auroc(y, s_boltz),
            "rankmean": _auroc(y, rankmean), "zsum": _auroc(y, zsum),
            "n_known": n_known, "n_random": n_rand,
        }
    scorers = ["smina", "boltz", "rankmean", "zsum"]
    mean = {s: float(np.mean([t[s] for t in per_target.values()])) if per_target else float("nan") for s in scorers}
    worst = {s: float(np.min([t[s] for t in per_target.values()])) if per_target else float("nan") for s in scorers}
    return {"per_target": per_target, "mean": mean, "worst": worst, "skipped": skipped}


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


@click.command()
@click.option("--dock-scores", default="data/dock/dock_scores.csv")
@click.option("--boltz", "boltz_csv", default="data/dock/sp_cs_boltz_controls.csv")
@click.option("--targets", default="O43570_WT,P10721_WT,P02753_WT,P0C559_WT")
@click.option("--min-known", default=5, type=int)
def main(dock_scores, boltz_csv, targets, min_known):
    tlist = [t.strip() for t in targets.split(",")]
    frame = build_frame(load_smina(dock_scores, tlist), load_boltz(boltz_csv))
    out = benchmark(frame, min_known=min_known)
    print("per-target AUROC (known vs random):")
    for t, e in out["per_target"].items():
        print(f"  {t:12} smina {e['smina']:.3f}  boltz {e['boltz']:.3f}  "
              f"rankmean {e['rankmean']:.3f}  zsum {e['zsum']:.3f}  (k={e['n_known']} r={e['n_random']})")
    print(f"skipped: {out['skipped']}")
    for agg in ("mean", "worst"):
        o = out[agg]
        print(f"{agg:6} smina {o['smina']:.3f}  boltz {o['boltz']:.3f}  "
              f"rankmean {o['rankmean']:.3f}  zsum {o['zsum']:.3f}")
    w = out["worst"]
    best_single = max(w["smina"], w["boltz"])
    print(f"\nWORST-CASE rescue: rankmean {w['rankmean']:.3f} vs best-single {best_single:.3f} "
          f"-> {'consensus more robust' if w['rankmean'] > best_single else 'no rescue'}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_consensus_score.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/consensus_score.py tests/test_consensus_score.py
git commit -m "feat(SP-CS): consensus benchmark (join smina+Boltz, rank/z consensus, per-target AUROC)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 2: Generate Boltz known/random (ops)

**Files:**
- Create: `data/boltz/matrix_inputs_sp_cs.json` (subset of `matrix_inputs_powered.json` — the 4 targets' proteins)

- [ ] **Step 1: Build the 4-target inputs JSON**

```bash
cd ~/pw && .venv/bin/python -c "
import json
d = json.load(open('data/boltz/matrix_inputs_powered.json'))
four = {'O43570_WT','P10721_WT','P02753_WT','P0C559_WT'}
d['proteins'] = [p for p in d['proteins'] if p['target_id'] in four]
json.dump(d, open('data/boltz/matrix_inputs_sp_cs.json','w'), indent=2)
print('proteins:', [p['target_id'] for p in d['proteins']])
"
```

- [ ] **Step 2: Run `boltz_controls` (detached, proxy set, ~few hours)**

```bash
cd ~/pw
setsid env CUDA_VISIBLE_DEVICES=0 BOLTZ=.venv-boltz/bin/boltz \
  https_proxy=http://proxy.unimaas.nl:3128/ http_proxy=http://proxy.unimaas.nl:3128/ \
  HTTPS_PROXY=http://proxy.unimaas.nl:3128/ HTTP_PROXY=http://proxy.unimaas.nl:3128/ \
  nohup .venv-boltz/bin/python -m scripts.boltz_controls \
  --dock-scores data/dock/dock_scores.csv --inputs data/boltz/matrix_inputs_sp_cs.json \
  --scores data/dock/sp_cs_boltz_controls.csv --out-dir boltz_out/sp_cs \
  --cap 10 --batch --no-kernels --accelerator gpu --batch-in boltz_batch_in_sp_cs \
  > logs/sp_cs_boltz.log 2>&1 </dev/null &
# monitor: grep -E "pending|parsed" logs/sp_cs_boltz.log ; watch for "Failed to process" (MSA/proxy issues)
```

- [ ] **Step 3: On completion, verify** the scores CSV has known+random rows for all 4 targets with finite `affinity_pred` (grep the "parsed N/N" line; `pandas` value_counts on `class`/`target`). Re-run is idempotent (skips scored cells).

---

### Task 3: Run benchmark + results doc + finish

**Files:**
- Create: `docs/SP_CS_RESULTS.md`
- Modify: `.superpowers/sdd/progress.md`

- [ ] **Step 1: Run the benchmark** on the real data:

```bash
cd ~/pw && .venv/bin/python -m scripts.consensus_score \
  --boltz data/dock/sp_cs_boltz_controls.csv \
  --targets O43570_WT,P10721_WT,P02753_WT,P0C559_WT
```

- [ ] **Step 2: Write `docs/SP_CS_RESULTS.md`** — question; method (smina∩Boltz, rank-mean + z-sum consensus, per-target AUROC); the per-target table + mean + worst-case; the verdict (does consensus's worst-case beat each single scorer's worst-case — is it more robust?); caveats (4 targets, P0C559 low-N knowns, random decoys imperfect, Boltz likely high-ceiling); reproduce commands.

- [ ] **Step 3: Honest verdict** — one of: (a) consensus more robust (worst-case rescue, even at mean parity); (b) no rescue (single scorer already dominates); (c) consensus hurts. State with numbers.

- [ ] **Step 4: Update the SDD ledger** with an SP-CS block, then **commit**:

```bash
git add docs/SP_CS_RESULTS.md
git commit -m "docs(SP-CS): consensus benchmark results + verdict

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

- [ ] **Step 5: Finish the branch** — use superpowers:finishing-a-development-branch (merge to `powered-specificity`, push to origin fork; box authoritative, reconcile remote first as in SP-L/SP-F).

---

## Self-review notes

- **Task 1 is the load-bearing code** and is fully TDD'd, including the worst-case-rescue property (the headline metric) via a constructed smina-fails target — so the metric is proven to respond before real data arrives.
- **Task 2 is the only compute** (Boltz ~70–80 cells) and pure reuse of `boltz_controls`; the proxy + `BOLTZ` override are the two gotchas, both in Global Constraints.
- **AUROC direction:** strength = `−score` for both scorers; verified in the test (knowns strong ⇒ AUROC 1.0).
- **Intersection join** means the benchmark uses only molecules Boltz actually scored (capped set); smina has more but they're dropped — logged implicitly by row counts.
