"""Tier-2 readout (the decisive calibration): does docked own-vs-mismatch selectivity track MEASURED
selectivity? For every compound measured on a pair of targets (A,B), correlate measured Delta-affinity
pChEMBL(A)-pChEMBL(B) against docked selectivity z_A - z_B (z per pocket over the docking set = the
common-reference per-pocket scale). Spearman overall, WITHIN-family (paralog — the valuable case), and
CROSS-family.

    .venv/bin/python -m scripts.tier2_analyze

Sign: higher pChEMBL(A) = binds A better; more-negative smina z_A = binds A better. So if docking tracks
selectivity, measured_delta and docked_delta are NEGATIVELY related -> we report rho(measured, -docked)
so POSITIVE = docking tracks measured selectivity.
"""
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

KIN = {"KIT", "JAK3", "CDK5"}
GPCR = {"5HT1A", "5HT2A", "A1R"}
NAME_TID = {"KIT": "P10721_WT", "JAK3": "P52333_WT", "CDK5": "Q00535_WT",
            "5HT1A": "P08908_WT", "5HT2A": "P28223_WT", "A1R": "P30542_WT"}
fam = lambda n: "kin" if n in KIN else "gpcr"


def boot_rho(x, y, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        r = spearmanr(x[idx], y[idx]).correlation
        if r == r:
            rs.append(r)
    return np.percentile(rs, 2.5), np.percentile(rs, 97.5)


def main():
    df = pd.read_csv("data/dock/tier2/dock_scores.csv")
    measured = json.load(open("data/dock/tier2/measured.json"))
    names = list(NAME_TID)

    # per-pocket z over the docking set (best/min score per molecule-pocket)
    best = df.groupby(["molecule", "pocket"], as_index=False).score.min()
    piv = best.pivot(index="molecule", columns="pocket", values="score")
    Z = (piv - piv.mean(axis=0)) / piv.std(axis=0)

    triples = []  # (stratum, measured_delta, docked_delta)
    for smi, e in measured.items():
        if smi not in Z.index:
            continue
        meas_names = [n for n in names if n in e]
        for a, b in itertools.combinations(meas_names, 2):
            za, zb = Z.at[smi, NAME_TID[a]], Z.at[smi, NAME_TID[b]]
            if not (np.isfinite(za) and np.isfinite(zb)):
                continue
            md = e[a] - e[b]          # measured: >0 => binds A better
            dd = za - zb              # docked: <0 => binds A better
            strat = "within-" + fam(a) if fam(a) == fam(b) else "cross"
            triples.append((strat, md, dd))
    t = pd.DataFrame(triples, columns=["stratum", "measured_delta", "docked_delta"])
    print(f"triples: {len(t)} (within-kin {sum((t.stratum=='within-kin'))}, "
          f"within-gpcr {sum(t.stratum=='within-gpcr')}, cross {sum(t.stratum=='cross')})\n")

    def report(sub, label):
        if len(sub) < 8:
            print(f"{label:14} n={len(sub):4d}  (too few)"); return
        # rho(measured, -docked): POSITIVE => docking tracks measured selectivity
        rho = spearmanr(sub.measured_delta, -sub.docked_delta).correlation
        lo, hi = boot_rho(sub.measured_delta.values, -sub.docked_delta.values)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        print(f"{label:14} n={len(sub):4d}  rho={rho:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  {sig}")

    print("=== docked selectivity vs MEASURED selectivity (Spearman; + = tracks) ===")
    report(t, "ALL")
    report(t[t.stratum == "within-kin"], "within-kinase")
    report(t[t.stratum == "within-gpcr"], "within-GPCR")
    report(t[t.stratum == "cross"], "cross-family")
    t.to_csv("data/dock/tier2/tier2_triples.csv", index=False)
    print("\nwrote data/dock/tier2/tier2_triples.csv")


if __name__ == "__main__":
    main()
