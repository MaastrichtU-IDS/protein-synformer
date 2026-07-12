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
