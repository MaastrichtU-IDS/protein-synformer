"""SP-DPO held-out readout: for each held-out target, compare the DPO'd model's raw-sample
own-preference against the base model's, on the shared panel (own + 10 train pockets), using
the unit-tested pure functions in ``scripts.dpo_eval``. No new metric logic here.

    .venv/bin/python -m scripts.dpo_eval_report \
        --scores data/dock/dpo/heldout/eval_scores.csv \
        --train-json data/dock/dpo/train10.json \
        --heldout-json data/dock/dpo/heldout4.json \
        --base-dir data/dock/dpo/heldout/base \
        --dpo-dir  data/dock/dpo/heldout/dpo \
        --out data/dock/dpo/heldout/dpo_eval_summary.json

Reports per held-out target: n_base/n_dpo mols scored, mean own-preference d for each pool
(d = mean(mismatch_score) - own_score; HIGHER = more own-preferring), the DPO-minus-base
difference with an unpaired bootstrap CI, and the joint-z variant (more-NEGATIVE dz = more
specific). Also a pooled-across-targets difference. Family labels are informational.
"""
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from scripts.dpo_eval import own_preference, two_sample_diff_ci, joint_z_own_preference

# informational family labels for the 4 held-out targets
FAMILY = {
    "O75716_WT": "kinase (STK16)",
    "P28223_WT": "GPCR (5-HT2A)",
    "P15090_WT": "lipid-binding (FABP4)",
    "P0C559_WT": "bacterial ATPase (gyraseB)",
}


def _smiset(path):
    p = Path(path)
    return set(l.strip() for l in p.read_text().splitlines() if l.strip()) if p.exists() else set()


@click.command()
@click.option("--scores", default="data/dock/dpo/heldout/eval_scores.csv")
@click.option("--train-json", default="data/dock/dpo/train10.json")
@click.option("--heldout-json", default="data/dock/dpo/heldout4.json")
@click.option("--base-dir", default="data/dock/dpo/heldout/base")
@click.option("--dpo-dir", default="data/dock/dpo/heldout/dpo")
@click.option("--out", default="data/dock/dpo/heldout/dpo_eval_summary.json")
def main(scores, train_json, heldout_json, base_dir, dpo_dir, out):
    df = pd.read_csv(scores)
    train_pockets = [t["target_id"] for t in json.load(open(train_json))]
    heldout = [t["target_id"] for t in json.load(open(heldout_json))]

    rows = []
    all_base_d, all_dpo_d = [], []
    for tid in heldout:
        base_set = _smiset(f"{base_dir}/{tid}.smi")
        dpo_set = _smiset(f"{dpo_dir}/{tid}.smi")
        d_base = own_preference(df, tid, train_pockets, smiles_subset=base_set)
        d_dpo = own_preference(df, tid, train_pockets, smiles_subset=dpo_set)
        bv, dv = list(d_base.values()), list(d_dpo.values())
        if not bv or not dv:
            print(f"{tid}: SKIP (n_base={len(bv)} n_dpo={len(dv)})", flush=True)
            continue
        diff, lo, hi = two_sample_diff_ci(dv, bv)   # a=DPO, b=base
        # joint-z robustness variant (more-negative dz = more specific)
        origin = {**{s: "base" for s in base_set}, **{s: "dpo" for s in dpo_set}}
        jz = joint_z_own_preference(df, tid, train_pockets, origin)
        jzb = list(jz["base"].values()); jzd = list(jz["dpo"].values())
        jdiff = float(np.mean(jzd) - np.mean(jzb)) if jzb and jzd else float("nan")
        all_base_d += bv; all_dpo_d += dv
        rec = {
            "target": tid, "family": FAMILY.get(tid, "?"),
            "n_base": len(bv), "n_dpo": len(dv),
            "mean_d_base": float(np.mean(bv)), "mean_d_dpo": float(np.mean(dv)),
            "diff_dpo_minus_base": diff, "ci_lo": lo, "ci_hi": hi,
            "sig": (lo > 0 or hi < 0),
            "jointz_mean_base": float(np.mean(jzb)) if jzb else None,
            "jointz_mean_dpo": float(np.mean(jzd)) if jzd else None,
            "jointz_diff": jdiff,
        }
        rows.append(rec)
        arrow = "DPO>base" if diff > 0 else "DPO<base"
        star = " *" if rec["sig"] else ""
        print(f"{tid} [{rec['family']}]: d_base={rec['mean_d_base']:.3f} d_dpo={rec['mean_d_dpo']:.3f} "
              f"diff={diff:+.3f} CI[{lo:+.3f},{hi:+.3f}] {arrow}{star} | jz_diff={jdiff:+.3f} "
              f"(n_base={len(bv)} n_dpo={len(dv)})", flush=True)

    pooled = None
    if all_base_d and all_dpo_d:
        pdiff, plo, phi = two_sample_diff_ci(all_dpo_d, all_base_d)
        pooled = {"n_base": len(all_base_d), "n_dpo": len(all_dpo_d),
                  "mean_d_base": float(np.mean(all_base_d)), "mean_d_dpo": float(np.mean(all_dpo_d)),
                  "diff_dpo_minus_base": pdiff, "ci_lo": plo, "ci_hi": phi, "sig": (plo > 0 or phi < 0)}
        print(f"\nPOOLED (all held-out): d_base={pooled['mean_d_base']:.3f} d_dpo={pooled['mean_d_dpo']:.3f} "
              f"diff={pdiff:+.3f} CI[{plo:+.3f},{phi:+.3f}] {'SIG' if pooled['sig'] else 'ns'}", flush=True)
    json.dump({"per_target": rows, "pooled": pooled}, open(out, "w"), indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
