# SP-AD ADMET Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Score generated molecule pools with `admet-ai` → per-molecule endpoint CSV + profile + an `admet_pass` guard; characterize the 41 pocket-candidate pools.

**Architecture:** Pure helpers (pool loading, `admet_pass`, `profile`) unit-tested in `.venv`; the `admet-ai` model call runs in the isolated `.venv-admet` inside `main()`. Spec: `docs/superpowers/specs/2026-07-12-admet-harness-design.md`.

## Global Constraints

- **admet-ai runs in `.venv-admet`** (isolated, on the share; torch 2.5 + chemprop). Unit tests run in `.venv`.
- **Import `admet_ai` INSIDE `main()`** so pure helpers unit-test without it.
- **Guard directionality MUST be verified** against a live `predict` output (known-safe drug vs known hERG-blocker) before finalizing `admet_pass` — hERG/DILI/ClinTox/Carcinogens are binary classifiers (higher raw prob = more toxic); HIA higher = better absorption.
- Commit only task files via explicit `git add <path>` — never `git add -A`.
- Commit footer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg`.

---

### Task 1: `scripts/admet_score.py`

**Files:** Create `scripts/admet_score.py`, `tests/test_admet_score.py`

**Interfaces:**
- `load_pool(paths: list[str]) -> list[str]` — read SMILES (one/line) from files and/or `<target>.txt` in dirs; strip blanks; dedup preserving order.
- `admet_pass(df: pd.DataFrame, tox_max: float = 0.5, hia_min: float = 0.5) -> pd.Series` — per-row bool: passes iff raw `hERG`,`DILI`,`ClinTox`,`Carcinogens_Lagunin` probs all `< tox_max` AND `HIA_Hou` `>= hia_min`. Missing columns → treated as not-failing for that endpoint (and logged by caller).
- `profile(df: pd.DataFrame, pass_series: pd.Series) -> dict` — `{"n", "pass_rate", per-endpoint favorable fraction for the critical set, median of a few key percentiles}`.
- CLI: `admet_score --pools <files/dirs csv> --out <endpoints.csv> --summary <profile.json>` — loads pool, runs `ADMETModel().predict`, writes per-molecule CSV (smiles + all endpoints + `admet_pass`), prints + writes profile.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_admet_score.py
import pandas as pd
from scripts.admet_score import load_pool, admet_pass, profile


def test_load_pool_dedups_files_and_dirs(tmp_path):
    f = tmp_path / "a.txt"; f.write_text("CCO\n\nCCO\nc1ccccc1\n")
    assert load_pool([str(f)]) == ["CCO", "c1ccccc1"]


def test_admet_pass_clean_passes_toxic_fails():
    df = pd.DataFrame({
        "hERG": [0.1, 0.9], "DILI": [0.2, 0.2], "ClinTox": [0.1, 0.1],
        "Carcinogens_Lagunin": [0.2, 0.2], "HIA_Hou": [0.9, 0.9],
    })
    p = admet_pass(df)
    assert bool(p.iloc[0]) is True     # clean
    assert bool(p.iloc[1]) is False    # high hERG


def test_admet_pass_low_hia_fails():
    df = pd.DataFrame({"hERG":[0.1],"DILI":[0.1],"ClinTox":[0.1],
                       "Carcinogens_Lagunin":[0.1],"HIA_Hou":[0.2]})
    assert bool(admet_pass(df).iloc[0]) is False


def test_profile_pass_rate():
    df = pd.DataFrame({"hERG":[0.1,0.9],"DILI":[0.1,0.1],"ClinTox":[0.1,0.1],
                       "Carcinogens_Lagunin":[0.1,0.1],"HIA_Hou":[0.9,0.9]})
    ps = admet_pass(df)
    prof = profile(df, ps)
    assert prof["n"] == 2 and abs(prof["pass_rate"] - 0.5) < 1e-9
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest tests/test_admet_score.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Implement** (pure helpers + CLI with `admet_ai` imported inside `main`):

```python
# scripts/admet_score.py
"""Score generated molecule pools with admet-ai. Pure helpers unit-test in .venv;
the model call runs in .venv-admet (import inside main)."""
from __future__ import annotations
import json, pathlib
import click
import pandas as pd

CRITICAL_TOX = ["hERG", "DILI", "ClinTox", "Carcinogens_Lagunin"]


def load_pool(paths):
    smis, seen = [], set()
    for p in paths:
        p = pathlib.Path(p)
        files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
        for f in files:
            for ln in f.read_text().splitlines():
                s = ln.strip()
                if s and s not in seen:
                    seen.add(s); smis.append(s)
    return smis


def admet_pass(df, tox_max: float = 0.5, hia_min: float = 0.5):
    ok = pd.Series(True, index=df.index)
    for c in CRITICAL_TOX:
        if c in df.columns:
            ok &= df[c] < tox_max
    if "HIA_Hou" in df.columns:
        ok &= df["HIA_Hou"] >= hia_min
    return ok


def profile(df, pass_series) -> dict:
    out = {"n": int(len(df)), "pass_rate": float(pass_series.mean()) if len(df) else float("nan")}
    for c in CRITICAL_TOX:
        if c in df.columns:
            out[f"favorable_{c}"] = float((df[c] < 0.5).mean())
    for c in df.columns:
        if c.endswith("_drugbank_approved_percentile"):
            out[f"median_{c}"] = float(df[c].median())
    return out


@click.command()
@click.option("--pools", required=True, help="comma-separated files/dirs of SMILES")
@click.option("--out", required=True, help="per-molecule endpoints CSV")
@click.option("--summary", required=True, help="profile JSON")
def main(pools, out, summary):
    from admet_ai import ADMETModel
    smis = load_pool([p.strip() for p in pools.split(",")])
    print(f"scoring {len(smis)} unique SMILES with admet-ai", flush=True)
    model = ADMETModel()
    df = model.predict(smiles=smis)
    df.insert(0, "smiles", smis if len(smis) == len(df) else df.index)
    ps = admet_pass(df)
    df["admet_pass"] = ps.values
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    prof = profile(df, ps)
    json.dump(prof, open(summary, "w"), indent=2)
    print(f"admet_pass rate: {prof['pass_rate']:.2%} of {prof['n']}", flush=True)
    for k, v in prof.items():
        if k.startswith("favorable_"):
            print(f"  {k}: {v:.2%}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify directionality on the box** (`.venv-admet`, 2-molecule probe): score a known-safe drug vs a known hERG-blocker (e.g. `CCO` vs `astemizole`/`terfenadine` SMILES); confirm the hERG raw column is higher for the blocker (so `< tox_max` = pass is correct). Adjust `CRITICAL_TOX` column names to the exact `predict` output if they differ (e.g. suffix). Document in report.

- [ ] **Step 5: Run tests** — `.venv/bin/python -m pytest tests/test_admet_score.py -q` (4 pass).

- [ ] **Step 6: Commit**

```bash
git add scripts/admet_score.py tests/test_admet_score.py
git commit -m "feat(SP-AD): admet_score.py — admet-ai wrapper + admet_pass guard + profile

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8AVKWfNxzdG4Db2onxnkg"
```

---

### Task 2: Characterize the 41 pocket-candidate pools + report

- [ ] **Step 1: Run** (`.venv-admet`, proxy in case of model re-download):

```bash
cd ~/pw && export https_proxy=http://proxy.unimaas.nl:3128/ HTTPS_PROXY=http://proxy.unimaas.nl:3128/
.venv-admet/bin/python -m scripts.admet_score --pools data/dock/candidates_pocket \
  --out data/dock/admet_candidates.csv --summary data/dock/admet_candidates_profile.json
```

- [ ] **Step 2: Write `docs/SP_AD_RESULTS.md`** — the generated pools' ADMET profile: overall `admet_pass` rate, per-critical-endpoint favorable %, key endpoint medians vs approved-drug percentiles (how drug-like/safe is what the generator produces?), and what the distributions imply for the DPO `admet_pass` thresholds. Reproduce commands.

- [ ] **Step 3: Update SDD ledger; commit** results doc. (Data CSVs stay on the share — gitignored.)

- [ ] **Step 4: Finish branch** — superpowers:finishing-a-development-branch (merge to `powered-specificity`, push to fork).

---

## Self-review notes
- Pure helpers (`load_pool`/`admet_pass`/`profile`) carry the tests; the `admet-ai` call is spike-verified integration.
- **Directionality verification (Task 1 Step 4) is load-bearing** — a flipped hERG direction would invert the guard; verify with a known blocker before trusting `admet_pass`.
- `.venv-admet` is isolated — no risk to other venvs.
