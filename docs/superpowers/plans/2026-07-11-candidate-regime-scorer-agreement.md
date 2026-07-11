# SP-CC Candidate-Regime Scorer-Agreement Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether smina and Boltz disagree more on generated candidates than on known/random molecules (the non-circular regime contrast), and whether smina's top candidates are Boltz-outliers (hacking), using a stratified Boltz sample of the SP-C pocket candidates.

**Architecture:** Pure analysis + one modest Boltz run. A deterministic stratified sampler picks ~30 candidates/target across the smina range; `boltz_controls._run_batch` co-folds them; a pure analysis module computes per-target Spearman (candidate vs known/random regime), the smina-top hacking percentile, and selection overlap.

**Tech Stack:** Python 3.10, pandas, numpy, scipy (`spearmanr`, confirmed present), click, pytest; Boltz-2 via `.venv-boltz`. Spec: `docs/superpowers/specs/2026-07-11-candidate-regime-scorer-agreement-design.md`.

## Global Constraints

- **Analysis + tests in `.venv`** (`.venv/bin/python -m pytest`). Boltz run in `.venv-boltz`.
- **Boltz needs the proxy:** `https_proxy=http://proxy.unimaas.nl:3128/` (and HTTP*/HTTPS* variants) or all MSA requests fail; `BOLTZ=.venv-boltz/bin/boltz`; `CUDA_VISIBLE_DEVICES=0`; `--no-kernels`.
- **5 targets:** `O43570_WT, P06537_WT, P10721_WT, P02753_WT, P0C559_WT`. (P06537 known/random regime is low-N — 3 knowns; its candidate-regime point is fine.)
- **strength = −score** for both scorers (lower score = stronger). Spearman over strengths.
- **Headline = regime contrast** (candidate vs known/random Spearman), non-circular. Hacking + selection-overlap are Boltz-referenced ⇒ **illustration only**.
- **Commits:** author `michel.dumontier@gmail.com`; footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session:` line. Commit only task files via explicit `git add <path>` — never `git add -A`.
- **No new docking:** candidate smina is in `dock_scores_pocket.csv` (own-pocket, `source=='candidate'`, 150/target).

---

### Task 1: `candidate_boltz.py` — stratified sampler + Boltz-gen CLI

**Files:**
- Create: `scripts/candidate_boltz.py`
- Test: `tests/test_candidate_boltz.py`

**Interfaces:**
- Consumes: `pandas`; `scripts.boltz_controls` (`_run_batch`, `stem_for`) inside `main()` only.
- Produces:
  - `load_candidates(dock_scores_pocket_csv, targets) -> pd.DataFrame` cols `[target, molecule, smina]` — rows with `pocket==target`, `source=='candidate'`, target in `targets`; `smina = score`.
  - `stratified_sample(df, n_per_target=30, strata=3) -> pd.DataFrame` — per target, sort by `smina` ascending (strongest first), split into `strata` equal contiguous bins, take `n_per_target//strata` **evenly-spaced** rows from each bin (deterministic). Returns the sampled subset (same columns).
  - CLI `candidate_boltz --pocket-scores data/dock/dock_scores_pocket.csv --inputs data/boltz/matrix_inputs_powered.json --targets <csv> --n 30 --scores data/dock/sp_cc_candidate_boltz.csv --out-dir boltz_out/sp_cc --batch-in boltz_batch_in_sp_cc` — samples, builds cells `{target, class:"candidate", smiles, sequence, stem}`, calls `_run_batch(..., "gpu", True, batch_in)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_candidate_boltz.py
import pandas as pd
from scripts.candidate_boltz import stratified_sample


def test_stratified_sample_is_deterministic_and_spans_range():
    # one target, 30 candidates smina -15..+14 (strong..weak)
    df = pd.DataFrame({"target": ["T"] * 30, "molecule": [f"m{i}" for i in range(30)],
                       "smina": [float(-15 + i) for i in range(30)]})
    a = stratified_sample(df, n_per_target=6, strata=3)
    b = stratified_sample(df, n_per_target=6, strata=3)
    assert list(a.molecule) == list(b.molecule)              # deterministic
    assert len(a) == 6                                        # 2 per stratum x 3
    smi = sorted(a.smina.tolist())
    # spans strong / mid / weak: min in bottom third, max in top third
    assert smi[0] <= -10 and smi[-1] >= 9


def test_stratified_sample_per_target():
    df = pd.DataFrame({"target": ["A"] * 12 + ["B"] * 12,
                       "molecule": [f"a{i}" for i in range(12)] + [f"b{i}" for i in range(12)],
                       "smina": [float(i) for i in range(12)] * 2})
    s = stratified_sample(df, n_per_target=6, strata=3)
    assert set(s.target) == {"A", "B"}
    assert (s.target == "A").sum() == 6 and (s.target == "B").sum() == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_candidate_boltz.py -q`
Expected: FAIL (`ModuleNotFoundError: scripts.candidate_boltz`).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/candidate_boltz.py
"""Stratified-sample SP-C pocket candidates across the smina range and Boltz-score them
(reusing boltz_controls._run_batch), for the candidate-regime scorer-agreement benchmark."""
from __future__ import annotations

import json
import pathlib

import click
import numpy as np
import pandas as pd


def load_candidates(pocket_scores_csv, targets) -> pd.DataFrame:
    d = pd.read_csv(pocket_scores_csv)
    d = d[(d.target.isin(targets)) & (d.pocket == d.target) & (d.source == "candidate")]
    return (d[["target", "molecule", "score"]].rename(columns={"score": "smina"})
            .dropna(subset=["smina"]).drop_duplicates(["target", "molecule"]))


def _pick_evenly(idx: list[int], count: int) -> list[int]:
    if count >= len(idx):
        return idx
    pos = np.linspace(0, len(idx) - 1, count).round().astype(int)
    return [idx[p] for p in sorted(set(pos.tolist()))]


def stratified_sample(df, n_per_target: int = 30, strata: int = 3) -> pd.DataFrame:
    per_stratum = n_per_target // strata
    out = []
    for target, g in df.groupby("target"):
        g = g.sort_values("smina").reset_index(drop=True)
        bins = np.array_split(np.arange(len(g)), strata)
        chosen: list[int] = []
        for b in bins:
            chosen += _pick_evenly(list(b), per_stratum)
        out.append(g.iloc[sorted(set(chosen))])
    return pd.concat(out, ignore_index=True)


@click.command()
@click.option("--pocket-scores", default="data/dock/dock_scores_pocket.csv")
@click.option("--inputs", default="data/boltz/matrix_inputs_powered.json")
@click.option("--targets", default="O43570_WT,P06537_WT,P10721_WT,P02753_WT,P0C559_WT")
@click.option("--n", default=30, type=int)
@click.option("--scores", default="data/dock/sp_cc_candidate_boltz.csv")
@click.option("--out-dir", default="boltz_out/sp_cc")
@click.option("--batch-in", default="boltz_batch_in_sp_cc")
@click.option("--samples", default=3, type=int)
def main(pocket_scores, inputs, targets, n, scores, out_dir, batch_in, samples):
    from scripts.boltz_controls import _run_batch, stem_for

    tlist = [t.strip() for t in targets.split(",")]
    seq_of = {p["target_id"]: p["sequence"] for p in json.load(open(inputs))["proteins"]}
    sample = stratified_sample(load_candidates(pocket_scores, tlist), n_per_target=n)
    cells = [{"target": r.target, "class": "candidate", "smiles": r.molecule,
              "sequence": seq_of[r.target], "stem": stem_for(r.target, "candidate", r.molecule)}
             for r in sample.itertuples() if r.target in seq_of]
    pathlib.Path(scores).parent.mkdir(parents=True, exist_ok=True)
    print(f"sampled {len(cells)} candidates across {len(tlist)} targets -> Boltz", flush=True)
    _run_batch(cells, out_dir, scores, samples, "gpu", True, batch_in)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_candidate_boltz.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/candidate_boltz.py tests/test_candidate_boltz.py
git commit -m "feat(SP-CC): stratified candidate sampler + Boltz-gen CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 2: `candidate_agreement.py` — Spearman, hacking, overlap

**Files:**
- Create: `scripts/candidate_agreement.py`
- Test: `tests/test_candidate_agreement.py`

**Interfaces:**
- Consumes: `pandas`, `numpy`, `scipy.stats.spearmanr`; `scripts.consensus_score` (`load_smina`) for the known/random regime frame.
- Produces:
  - `spearman_by_target(frame) -> dict[str,float]` — per target, `spearmanr(−smina, −boltz)` (strengths) over that target's rows; `nan` if <3 rows.
  - `hacking_percentile(frame, k=5) -> dict[str,float]` — per target, take the k strongest-by-smina rows; return the mean percentile of their **boltz strength** within the target's rows (0–1; low ⇒ smina-top are Boltz-weak).
  - `selection_overlap(frame, k=5) -> dict[str,dict]` — per target, Jaccard of smina-top-k vs boltz-top-k, and vs consensus(rank-mean)-top-k.
  - `regime_contrast(candidate_frame, knownrandom_frame) -> dict` — `spearman_by_target` on each; per-target `{cand, known_random}` + mean of each.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_candidate_agreement.py
import numpy as np
import pandas as pd
from scripts.candidate_agreement import spearman_by_target, hacking_percentile, selection_overlap


def test_spearman_agree_vs_disagree():
    agree = pd.DataFrame({"target": ["A"] * 5, "molecule": list("abcde"),
                          "smina": [-9, -7, -5, -3, -1], "boltz": [-9, -7, -5, -3, -1]})
    disagree = agree.copy(); disagree["boltz"] = [-1, -3, -5, -7, -9]  # inverted
    assert spearman_by_target(agree)["A"] > 0.99
    assert spearman_by_target(disagree)["A"] < -0.99


def test_hacking_percentile_low_when_smina_top_are_boltz_weak():
    # smina-strongest (m0,m1) are Boltz-weakest -> low percentile
    df = pd.DataFrame({"target": ["A"] * 5, "molecule": [f"m{i}" for i in range(5)],
                       "smina": [-9, -8, -5, -3, -1], "boltz": [-1, -2, -5, -8, -9]})
    p = hacking_percentile(df, k=2)["A"]
    assert p < 0.4          # smina-top are near the bottom of Boltz


def test_selection_overlap_jaccard():
    # smina and boltz pick disjoint top-2 -> jaccard 0
    df = pd.DataFrame({"target": ["A"] * 4, "molecule": list("abcd"),
                       "smina": [-9, -8, -2, -1], "boltz": [-1, -2, -8, -9]})
    o = selection_overlap(df, k=2)["A"]
    assert o["smina_vs_boltz"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_candidate_agreement.py -q`
Expected: FAIL (`ModuleNotFoundError: scripts.candidate_agreement`).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/candidate_agreement.py
"""Candidate-regime scorer-agreement analysis: per-target smina<->Boltz Spearman (candidate vs
known/random regime), smina-top hacking percentile, and selection overlap. Headline = regime contrast."""
from __future__ import annotations

import click
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _load_candidate_boltz(csv) -> pd.DataFrame:
    return (pd.read_csv(csv).rename(columns={"smiles": "molecule", "affinity_pred": "boltz"})
            [["target", "molecule", "boltz"]].dropna(subset=["boltz"]))


def spearman_by_target(frame) -> dict:
    out = {}
    for target, g in frame.groupby("target"):
        if len(g) < 3:
            out[str(target)] = float("nan"); continue
        rho = spearmanr(-g.smina.to_numpy(float), -g.boltz.to_numpy(float)).correlation
        out[str(target)] = float(rho)
    return out


def _pct_rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def hacking_percentile(frame, k: int = 5) -> dict:
    out = {}
    for target, g in frame.groupby("target"):
        g = g.copy()
        g["boltz_pct"] = _pct_rank(-g.boltz.to_numpy(float))        # 1 = strongest boltz
        top = g.sort_values("smina").head(k)                        # k strongest by smina
        out[str(target)] = float(top["boltz_pct"].mean())
    return out


def _topk(g, col, k):
    return set(g.sort_values(col).head(k).molecule) if col != "rankmean" else \
        set(g.assign(rankmean=(pd.Series(-g.smina.to_numpy(float)).rank(ascending=False).to_numpy()
                               + pd.Series(-g.boltz.to_numpy(float)).rank(ascending=False).to_numpy()))
            .sort_values("rankmean").head(k).molecule)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def selection_overlap(frame, k: int = 5) -> dict:
    out = {}
    for target, g in frame.groupby("target"):
        s = set(g.sort_values("smina").head(k).molecule)
        b = set(g.sort_values("boltz").head(k).molecule)
        c = _topk(g, "rankmean", k)
        out[str(target)] = {"smina_vs_boltz": _jaccard(s, b),
                            "consensus_vs_smina": _jaccard(c, s),
                            "consensus_vs_boltz": _jaccard(c, b)}
    return out


def regime_contrast(candidate_frame, knownrandom_frame) -> dict:
    cand = spearman_by_target(candidate_frame)
    kr = spearman_by_target(knownrandom_frame)
    per = {t: {"candidate": cand.get(t, float("nan")), "known_random": kr.get(t, float("nan"))}
           for t in set(cand) | set(kr)}
    finite = lambda d: [v for v in d.values() if v == v]
    return {"per_target": per,
            "mean_candidate": float(np.mean(finite(cand))) if finite(cand) else float("nan"),
            "mean_known_random": float(np.mean(finite(kr))) if finite(kr) else float("nan")}


@click.command()
@click.option("--candidate-boltz", default="data/dock/sp_cc_candidate_boltz.csv")
@click.option("--pocket-scores", default="data/dock/dock_scores_pocket.csv")
@click.option("--kr-boltz", default="data/dock/sp_cs_boltz_controls.csv")
@click.option("--dock-scores", default="data/dock/dock_scores.csv")
@click.option("--targets", default="O43570_WT,P06537_WT,P10721_WT,P02753_WT,P0C559_WT")
def main(candidate_boltz, pocket_scores, kr_boltz, dock_scores, targets):
    from scripts.candidate_boltz import load_candidates
    from scripts.consensus_score import load_smina, load_boltz, build_frame

    tlist = [t.strip() for t in targets.split(",")]
    cand = load_candidates(pocket_scores, tlist).merge(_load_candidate_boltz(candidate_boltz),
                                                       on=["target", "molecule"], how="inner")
    kr = build_frame(load_smina(dock_scores, tlist), load_boltz(kr_boltz))  # known/random regime
    rc = regime_contrast(cand, kr)
    print("REGIME CONTRAST — smina<->Boltz Spearman (higher = agree):")
    for t, e in sorted(rc["per_target"].items()):
        print(f"  {t:12} candidate {e['candidate']:+.3f}   known/random {e['known_random']:+.3f}")
    print(f"  MEAN         candidate {rc['mean_candidate']:+.3f}   known/random {rc['mean_known_random']:+.3f}")
    print("\nHACKING — mean Boltz percentile of smina-top-5 (low = smina-top are Boltz-weak):")
    for t, p in sorted(hacking_percentile(cand).items()):
        print(f"  {t:12} {p:.2f}")
    print("\nSELECTION OVERLAP (Jaccard, top-5):")
    for t, o in sorted(selection_overlap(cand).items()):
        print(f"  {t:12} smina~boltz {o['smina_vs_boltz']:.2f}  cons~smina {o['consensus_vs_smina']:.2f}  cons~boltz {o['consensus_vs_boltz']:.2f}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_candidate_agreement.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/candidate_agreement.py tests/test_candidate_agreement.py
git commit -m "feat(SP-CC): candidate-regime analysis (Spearman contrast, hacking pct, overlap)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 3: Generate Boltz + run analysis + results + finish

- [ ] **Step 1: Generate candidate Boltz** (detached, proxy set, ~few hours):

```bash
cd ~/pw
setsid env CUDA_VISIBLE_DEVICES=0 BOLTZ=.venv-boltz/bin/boltz \
  https_proxy=http://proxy.unimaas.nl:3128/ http_proxy=http://proxy.unimaas.nl:3128/ \
  HTTPS_PROXY=http://proxy.unimaas.nl:3128/ HTTP_PROXY=http://proxy.unimaas.nl:3128/ \
  nohup .venv-boltz/bin/python -m scripts.candidate_boltz \
  --targets O43570_WT,P06537_WT,P10721_WT,P02753_WT,P0C559_WT --n 30 \
  --scores data/dock/sp_cc_candidate_boltz.csv --out-dir boltz_out/sp_cc --batch-in boltz_batch_in_sp_cc \
  > logs/sp_cc_boltz.log 2>&1 </dev/null &
# monitor "parsed N/N"; watch for "Failed to process" (proxy/MSA)
```

- [ ] **Step 2: On completion, run the analysis:**

```bash
cd ~/pw && .venv/bin/python -m scripts.candidate_agreement \
  --candidate-boltz data/dock/sp_cc_candidate_boltz.csv \
  --kr-boltz data/dock/sp_cs_boltz_controls.csv
```

- [ ] **Step 3: Write `docs/SP_CC_RESULTS.md`** — the regime-contrast table (candidate vs known/random Spearman, per target + mean) as headline; hacking percentile + selection overlap as illustration; verdict (is candidate-regime disagreement > known/random?); caveats (N=5, 30/target, Boltz-referenced illustration is partly circular, P06537 known/random low-N). Reproduce commands.

- [ ] **Step 4: Honest verdict** — one of: (a) scorers disagree more on candidates than known/random (backs SP-CS: consensus/Boltz-validation matters specifically in the optimization regime); (b) no regime difference; (c) mixed. State with numbers.

- [ ] **Step 5: Update the SDD ledger; commit** results doc.

- [ ] **Step 6: Finish the branch** — superpowers:finishing-a-development-branch (merge to `powered-specificity`, push to fork; box authoritative, reconcile remote first as before).

---

## Self-review notes

- **Non-circular headline** (regime-contrast Spearman) is fully in the pure, TDD'd analysis (Task 2); the Boltz-referenced hacking/overlap metrics are clearly labeled illustration.
- **Determinism:** `stratified_sample` picks evenly-spaced indices within sorted thirds — reproducible; tested.
- **Only compute** is Task 3 Step 1 (~150 Boltz cells); reuses `boltz_controls._run_batch` + proxy/BOLTZ gotchas (Global Constraints).
- Candidate Boltz joins to smina on `(target, molecule)`; known/random regime reuses SP-CS data via `consensus_score.load_smina/load_boltz`.
