"""Tier-1 calibration readout: does the own-vs-mismatch z-delta (the metric the 'selection works'
positive used) separate REAL known actives from property-matched decoys — and where do the model's
generated candidates fall?

Builds ONE molecule x pocket score matrix pooling all three classes, z-normalizes each pocket column
across the pooled molecules (common per-pocket scale — the advisor's requirement), then per molecule:
    zdelta(m) = z(own source pocket) - mean(z(other panel pockets))
where MORE NEGATIVE = MORE own-preferring/specific (smina lower=better; matches _delta_win_from_matrix).
Compares the zdelta distribution across classes; AUROC(actives vs decoys) uses -zdelta as the score
(actives should be more own-preferring => more negative zdelta => higher -zdelta).

    .venv/bin/python -m scripts.tier1_analyze
"""
import json
import numpy as np
import pandas as pd
from scripts.dpo_eval import two_sample_diff_ci
from scripts.boltz_controls_analyze import discrimination_auroc


def load_all():
    frames = []
    for cls in ["actives", "decoys", "candidates"]:
        d = pd.read_csv(f"data/dock/tier1/{cls}_scores.csv")
        d["cls"] = cls
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def zdelta_by_molecule(df, pockets):
    """df has target(source), pocket, molecule, cls, score. Returns per-(cls,molecule) zdelta on the
    shared panel, z per pocket-column across ALL pooled molecules (best/min score per molecule-pocket)."""
    # best score per (molecule, pocket); keep source+cls
    best = (df.groupby(["cls", "target", "molecule", "pocket"], as_index=False)
              .score.min())
    # pivot molecule x pocket (index carries cls+target(source))
    piv = best.pivot_table(index=["cls", "target", "molecule"], columns="pocket", values="score")
    piv = piv.reindex(columns=pockets)
    # z per pocket column across all pooled molecules (nan-aware)
    Z = (piv - piv.mean(axis=0)) / piv.std(axis=0)
    rows = []
    for (cls, src, mol), zr in Z.iterrows():
        own = zr.get(src, np.nan)
        offs = zr.drop(labels=[src]).dropna().values
        if not np.isfinite(own) or len(offs) == 0:
            continue
        rows.append({"cls": cls, "target": src, "molecule": mol,
                     "zdelta": float(own - offs.mean())})
    return pd.DataFrame(rows)


def main():
    df = load_all()
    pockets = [t["target_id"] for t in json.load(open("data/dock/tier1/panel8.json"))]
    zd = zdelta_by_molecule(df, pockets)
    print("n by class:\n", zd.groupby("cls").zdelta.agg(["count", "mean", "median"]).to_string(), "\n")

    a = zd[zd.cls == "actives"].zdelta.values
    d = zd[zd.cls == "decoys"].zdelta.values
    c = zd[zd.cls == "candidates"].zdelta.values

    # actives more own-preferring than decoys => actives more NEGATIVE zdelta => diff(actives-decoys) < 0
    diff_ad, lo_ad, hi_ad = two_sample_diff_ci(a, d)
    diff_ac, lo_ac, hi_ac = two_sample_diff_ci(a, c)
    # AUROC: can the zdelta rank actives above decoys? higher -zdelta = more own-preferring = active-like
    auroc_ad = discrimination_auroc(
        pd.DataFrame({"class": (["known"] * len(a)) + (["random"] * len(d)),
                      "val": list(-a) + list(-d)}), "val", higher_is_better=True)

    print(f"actives vs decoys : mean zdelta {a.mean():+.3f} vs {d.mean():+.3f} | "
          f"diff(a-d)={diff_ad:+.3f} CI[{lo_ad:+.3f},{hi_ad:+.3f}] "
          f"{'(actives MORE own-preferring, sig)' if hi_ad < 0 else '(ns/!)'}")
    print(f"                    AUROC(zdelta ranks actives>decoys) = {auroc_ad:.3f} "
          f"(0.5=metric blind to real-vs-decoy own-preference)")
    print(f"actives vs candidates: mean zdelta {a.mean():+.3f} vs {c.mean():+.3f} | "
          f"diff(a-c)={diff_ac:+.3f} CI[{lo_ac:+.3f},{hi_ac:+.3f}] "
          f"{'(candidates own-prefer >= actives => generation artifact risk)' if diff_ac <= 0 or lo_ac <= 0 <= hi_ac else ''}")

    out = {"n": zd.groupby("cls").size().to_dict(),
           "mean_zdelta": zd.groupby("cls").zdelta.mean().to_dict(),
           "actives_vs_decoys": {"diff": diff_ad, "ci": [lo_ad, hi_ad], "auroc": auroc_ad},
           "actives_vs_candidates": {"diff": diff_ac, "ci": [lo_ac, hi_ac]}}
    json.dump(out, open("data/dock/tier1/tier1_summary.json", "w"), indent=2)
    print("\nwrote data/dock/tier1/tier1_summary.json")


if __name__ == "__main__":
    main()
