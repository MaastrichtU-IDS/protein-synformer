"""Analyze the Boltz-2 discrimination control: can Boltz separate known binders from
random molecules? Per target + pooled AUROC (known=positive) on affinity_pred (lower=
better) and binder_prob (higher=better), plus class means. AUROC >> 0.5 => Boltz
discriminates (the mismatch null is informative); ~0.5 => Boltz is blind here."""
from __future__ import annotations

import click
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def discrimination_auroc(df: pd.DataFrame, metric: str, higher_is_better: bool):
    """AUROC for ranking known (positive) above random by `metric`.
    Returns None if fewer than one finite value in either class (AUROC undefined)."""
    sub = df[["class", metric]].dropna()
    sub = sub[sub["class"].isin(["known", "random"])]
    y = (sub["class"] == "known").astype(int).to_numpy()
    if y.sum() < 1 or (1 - y).sum() < 1:
        return None
    score = sub[metric].to_numpy()
    if not higher_is_better:
        score = -score  # lower metric = stronger binder = more "known"
    return float(roc_auc_score(y, score))


@click.command()
@click.option("--scores", default="data/boltz/boltz_controls_scores.csv")
@click.option("--out", default="data/boltz/boltz_controls_summary.csv")
def main(scores, out):
    df = pd.read_csv(scores)
    targets = list(dict.fromkeys(df["target"]))
    rows = []
    print(f"{'target':12} {'n_kn':>5} {'n_rnd':>6} {'kn_aff':>8} {'rnd_aff':>8} "
          f"{'AUROC_aff':>10} {'AUROC_prob':>11}")
    for t in targets + ["POOLED"]:
        d = df if t == "POOLED" else df[df.target == t]
        kn = d[d["class"] == "known"]
        rnd = d[d["class"] == "random"]
        auroc_aff = discrimination_auroc(d, "affinity_pred", higher_is_better=False)
        auroc_prob = discrimination_auroc(d, "binder_prob", higher_is_better=True)
        row = {
            "target": t, "n_known": len(kn), "n_random": len(rnd),
            "known_aff_mean": float(np.nanmean(kn["affinity_pred"])) if len(kn) else float("nan"),
            "random_aff_mean": float(np.nanmean(rnd["affinity_pred"])) if len(rnd) else float("nan"),
            "known_prob_mean": float(np.nanmean(kn["binder_prob"])) if len(kn) else float("nan"),
            "random_prob_mean": float(np.nanmean(rnd["binder_prob"])) if len(rnd) else float("nan"),
            "auroc_affinity": auroc_aff, "auroc_binderprob": auroc_prob,
        }
        rows.append(row)
        fa = f"{auroc_aff:.3f}" if auroc_aff is not None else "  n/a"
        fp = f"{auroc_prob:.3f}" if auroc_prob is not None else "  n/a"
        print(f"{t:12} {len(kn):5d} {len(rnd):6d} {row['known_aff_mean']:8.2f} "
              f"{row['random_aff_mean']:8.2f} {fa:>10} {fp:>11}")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n(AUROC: known=positive; >0.5 = Boltz ranks known binders above random. "
          f"affinity lower=better, binderprob higher=better)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
