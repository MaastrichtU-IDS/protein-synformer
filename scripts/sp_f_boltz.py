"""SP-F Boltz-2 corroboration: does co-folding agree that the treatment arm's
top-M binders are stronger than control_b's, for the shakedown targets?

Recomputes the true top-M per (target, arm) from the union of that arm's round
dock_scores.csv (works around the final_topM.smi under-sizing), co-folds each into
the target's own sequence via Boltz-2 (reusing scripts.boltz_controls._run_batch),
and compares affinity_pred (lower = stronger). Run in .venv-boltz.
"""
from __future__ import annotations

import glob
import json
import pathlib

import click
import pandas as pd


def top_m_from_dock(round_csvs: list[str], m: int) -> list[str]:
    """Union the per-round dock_scores.csv, keep each SMILES' strongest (most-negative)
    score, return the m strongest SMILES."""
    best: dict[str, float] = {}
    for c in round_csvs:
        df = pd.read_csv(c)
        for smi, sc in zip(df.smiles, df.score):
            sc = float(sc)
            if smi not in best or sc < best[smi]:
                best[smi] = sc
    return [s for s, _ in sorted(best.items(), key=lambda kv: kv[1])[:m]]


def build_cells(topm_by_arm: dict[str, list[str]], target: str, seq: str) -> list[dict]:
    from scripts.boltz_controls import stem_for
    cells = []
    for arm, smis in topm_by_arm.items():
        for smi in smis:
            cells.append({"target": target, "class": arm, "smiles": smi,
                          "sequence": seq, "stem": stem_for(target, arm, smi)})
    return cells


def compare_boltz(df: pd.DataFrame) -> dict:
    """Per target, mean & best affinity_pred for each arm (lower = stronger) and the
    treatment-minus-control_b delta (negative ⇒ Boltz corroborates the docking win)."""
    out: dict[str, dict] = {}
    for target, g in df.groupby("target"):
        arms = {}
        for arm, ga in g.groupby("class"):
            aff = ga["affinity_pred"].astype(float)
            arms[arm] = {"mean_aff": float(aff.mean()), "best_aff": float(aff.min()), "n": int(len(ga))}
        e = dict(arms)
        if "treatment" in arms and "control_b" in arms:
            e["delta_mean_treatment_minus_control_b"] = arms["treatment"]["mean_aff"] - arms["control_b"]["mean_aff"]
        out[str(target)] = e
    return out


@click.command()
@click.option("--targets", default="O43570_WT,P06537_WT")
@click.option("--arms", default="treatment,control_b")
@click.option("--sp-f-dir", default="data/dock/sp_f")
@click.option("--inputs", default="data/boltz/matrix_inputs_powered.json")
@click.option("--m", default=10, type=int)
@click.option("--out-dir", default="boltz_out/sp_f")
@click.option("--scores", default="data/dock/sp_f_boltz_scores.csv")
@click.option("--samples", default=3, type=int)
@click.option("--batch-in", default="boltz_batch_in_sp_f")
def main(targets, arms, sp_f_dir, inputs, m, out_dir, scores, samples, batch_in):
    from scripts.boltz_controls import _run_batch

    seq_of = {p["target_id"]: p["sequence"] for p in json.load(open(inputs))["proteins"]}
    tlist = [t.strip() for t in targets.split(",")]
    alist = [a.strip() for a in arms.split(",")]
    cells: list[dict] = []
    for t in tlist:
        topm_by_arm = {}
        for arm in alist:
            csvs = glob.glob(f"{sp_f_dir}/{t}/{arm}/round_*/dock_scores.csv")
            topm_by_arm[arm] = top_m_from_dock(csvs, m)
            print(f"{t}/{arm}: top-{len(topm_by_arm[arm])} recomputed from {len(csvs)} round CSVs", flush=True)
        cells += build_cells(topm_by_arm, t, seq_of[t])
    pathlib.Path(scores).parent.mkdir(parents=True, exist_ok=True)
    _run_batch(cells, out_dir, scores, samples, "gpu", True, batch_in)

    df = pd.read_csv(scores)
    for target, e in compare_boltz(df).items():
        print(f"\n=== {target} (Boltz affinity_pred, lower=stronger) ===")
        for arm in alist:
            if arm in e:
                print(f"  {arm:10} mean {e[arm]['mean_aff']:.3f}  best {e[arm]['best_aff']:.3f}  n={e[arm]['n']}")
        if "delta_mean_treatment_minus_control_b" in e:
            d = e["delta_mean_treatment_minus_control_b"]
            print(f"  delta mean (treatment - control_b): {d:+.3f}  "
                  f"({'corroborates' if d < 0 else 'CONTRADICTS'} docking win)")


if __name__ == "__main__":
    main()
