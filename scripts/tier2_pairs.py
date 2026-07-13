"""Tier-2 step 2: from cached ChEMBL activities, find compounds measured on >=2 of the calibration
targets (the raw material for measured-selectivity calibration), report cross-target overlap, and write
(a) the docking set (union of overlap compounds, capped) and (b) a measured table keyed by SMILES.

    .venv/bin/python -m scripts.tier2_pairs [--cap 250]

Writes data/dock/tier2/dock_set.txt (SMILES/line) and data/dock/tier2/measured.json
({smiles: {chembl_id, per-target pChEMBL}}).
"""
import json
import itertools
from pathlib import Path

import click

KIN = {"KIT", "JAK3", "CDK5"}
GPCR = {"5HT1A", "5HT2A", "A1R"}
TARGET_TID = {"KIT": "P10721_WT", "JAK3": "P52333_WT", "CDK5": "Q00535_WT",
              "5HT1A": "P08908_WT", "5HT2A": "P28223_WT", "A1R": "P30542_WT"}
RAW = Path("data/dock/tier2/raw")


def fam(name):
    return "kin" if name in KIN else "gpcr"


@click.command()
@click.option("--cap", default=250, type=int, help="max compounds to dock (prefer measured-on-most-targets)")
def main(cap):
    names = list(TARGET_TID)
    per_target = {}
    for n in names:
        d = json.loads((RAW / f"{n}.json").read_text())["compounds"]
        per_target[n] = d
        print(f"{n}: {len(d)} compounds", flush=True)

    # smiles -> {chembl_id, target: pchembl}
    by_smi = {}
    for n in names:
        for mid, rec in per_target[n].items():
            e = by_smi.setdefault(rec["smiles"], {"chembl_id": mid})
            e[n] = rec["pchembl"]

    overlap = {s: e for s, e in by_smi.items() if sum(1 for n in names if n in e) >= 2}
    print(f"\ncompounds measured on >=2 targets: {len(overlap)}", flush=True)

    print("\n=== pairwise overlap (compounds with pChEMBL on BOTH) ===", flush=True)
    for a, b in itertools.combinations(names, 2):
        ov = sum(1 for e in by_smi.values() if a in e and b in e)
        tag = "within-" + fam(a) if fam(a) == fam(b) else "CROSS"
        if ov >= 5:
            print(f"  {a:6}/{b:6} [{tag:12}]: {ov}", flush=True)

    # docking set: prefer compounds measured on most targets
    ranked = sorted(overlap.items(), key=lambda kv: -sum(1 for n in names if n in kv[1]))
    chosen = dict(ranked[:cap])
    base = Path("data/dock/tier2")
    (base / "dock_set.txt").write_text("\n".join(chosen) + "\n")
    json.dump({s: e for s, e in chosen.items()}, open(base / "measured.json", "w"), indent=1)
    n_within = sum(1 for e in chosen.values()
                   if any(a in e and b in e for a, b in itertools.combinations(KIN, 2))
                   or any(a in e and b in e for a, b in itertools.combinations(GPCR, 2)))
    print(f"\ndocking set: {len(chosen)} compounds (>=1 within-family pair for {n_within}); "
          f"wrote {base/'dock_set.txt'} + measured.json", flush=True)
    print("TIER2 PAIRS DONE", flush=True)


if __name__ == "__main__":
    main()
