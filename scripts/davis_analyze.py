"""Tier-3 readout: does docked selectivity track DAVIS MEASURED selectivity across many kinase pairs?
Reuses the Tier-2 metric/convention (per-pocket z; ρ(measured, −docked) so + = docking tracks selectivity).

    .venv/bin/python -m scripts.davis_analyze
"""
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROTEIN_KINASES = {"KIT", "JAK3", "FGFR1", "CDK5", "DYRK1A", "CSNK1A1", "CSNK1G1", "CSNK1E",
                   "PHKG1", "STK16", "NEK1", "CAMK4", "DAPK2"}  # primary; PIK3CD/RIOK1 = robustness only


def summarize_pairs(per_pair_rho: dict) -> dict:
    vals = [v for v in per_pair_rho.values() if v == v]
    return {"n_pairs": len(vals), "n_positive": int(sum(v > 0 for v in vals)),
            "median_rho": float(np.median(vals)) if vals else float("nan")}


def _triples(measured, Z, tid, genes):
    """List of (drug, gA, gB, measured_delta, docked_delta) over gene pairs within `genes`."""
    out = []
    for smi, m in measured.items():
        if smi not in Z.index:
            continue
        gs = [g for g in genes if g in m]
        for a, b in itertools.combinations(gs, 2):
            za, zb = Z.at[smi, tid[a]], Z.at[smi, tid[b]]
            if np.isfinite(za) and np.isfinite(zb):
                out.append((smi, a, b, m[a] - m[b], za - zb))
    return out


def _pooled_rho(trips):
    md = [t[3] for t in trips]
    dd = [-t[4] for t in trips]
    return spearmanr(md, dd).correlation


def _clustered_ci(trips, nb=5000, seed=42):
    rng = np.random.default_rng(seed)
    by_drug = {}
    for t in trips:
        by_drug.setdefault(t[0], []).append(t)
    drugs = list(by_drug)
    rs = []
    for _ in range(nb):
        samp = rng.choice(drugs, len(drugs), replace=True)
        pool = [t for d in samp for t in by_drug[d]]
        if len(pool) > 5:
            r = _pooled_rho(pool)
            if r == r:
                rs.append(r)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def _report(measured, Z, tid, genes, label):
    trips = _triples(measured, Z, tid, genes)
    per_pair = {}
    for (a, b) in itertools.combinations([g for g in genes], 2):
        sub = [t for t in trips if t[1] == a and t[2] == b]
        if len(sub) >= 8:
            per_pair[(a, b)] = spearmanr([t[3] for t in sub], [-t[4] for t in sub]).correlation
    pooled = _pooled_rho(trips)
    lo, hi = _clustered_ci(trips)
    s = summarize_pairs(per_pair)
    sig = "SIG" if (lo > 0 or hi < 0) else "ns"
    print(f"{label}: {len(genes)} kinases, {s['n_pairs']} pairs w/>=8 drugs, {len(trips)} triples | "
          f"pooled rho={pooled:+.3f} clustered CI[{lo:+.3f},{hi:+.3f}] {sig} | "
          f"per-pair: {s['n_positive']}/{s['n_pairs']} positive, median {s['median_rho']:+.3f}")
    return {"label": label, "n_kinases": len(genes), "n_triples": len(trips),
            "pooled_rho": pooled, "ci": [lo, hi], "sig": (lo > 0 or hi < 0), **s}


def main():
    df = pd.read_csv("data/dock/davis/dock_scores.csv")
    measured = json.load(open("data/dock/davis/measured_davis.json"))
    tid = json.load(open("data/dock/davis/kinase_pockets.json"))  # gene -> target_id
    best = df.groupby(["molecule", "pocket"], as_index=False).score.min()
    piv = best.pivot(index="molecule", columns="pocket", values="score")
    Z = (piv - piv.mean(axis=0)) / piv.std(axis=0)
    all_genes = [g for g in tid if tid[g] in Z.columns]
    prot = [g for g in all_genes if g in PROTEIN_KINASES]
    print("=== DAVIS docked vs measured selectivity (Spearman; + = docking tracks; vs Tier-2 0.245) ===")
    out = {"PRIMARY_protein_kinases": _report(measured, Z, tid, prot, "PRIMARY (protein kinases)"),
           "ROBUSTNESS_all": _report(measured, Z, tid, all_genes, "ROBUSTNESS (all)")}
    json.dump(out, open("data/dock/davis/davis_summary.json", "w"), indent=2)
    print("\nwrote data/dock/davis/davis_summary.json")


if __name__ == "__main__":
    main()
