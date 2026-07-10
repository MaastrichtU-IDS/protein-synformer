"""SP-F readout: three-way arm decomposition + docking-budget parity check.

Reads a fragment_loop `loop_summary.csv` and reports, per target, each arm's
final-round top-10 mean docking score and best-overall binder, plus the pairwise
deltas that decompose the effect:
  treatment - control_a  = value of docking-guided seed choice
  control_a - control_b  = value of the analog mechanism itself
  treatment - control_b  = total loop value
(more negative = stronger binding = better). Also verifies the arms docked the
same number of molecules per round (budget parity — the SP-L lesson); a broken
parity invalidates a naive arm comparison.
"""
from __future__ import annotations

import click
import pandas as pd


def compare_arms(df: pd.DataFrame) -> dict:
    per_target: dict[str, dict] = {}
    for target, g in df.groupby("target"):
        arms: dict[str, dict] = {}
        for arm, ga in g.groupby("arm"):
            ga = ga.sort_values("round")
            arms[arm] = {
                "final_top10_mean": float(ga.iloc[-1]["top10_mean"]),
                "best_overall": float(ga["best"].min()),
            }
        entry: dict = dict(arms)

        def fm(a: str) -> float:
            return arms[a]["final_top10_mean"]

        if "treatment" in arms and "control_a" in arms:
            entry["delta_treatment_minus_control_a"] = fm("treatment") - fm("control_a")
        if "control_a" in arms and "control_b" in arms:
            entry["delta_control_a_minus_control_b"] = fm("control_a") - fm("control_b")
        if "treatment" in arms and "control_b" in arms:
            entry["delta_treatment_minus_control_b"] = fm("treatment") - fm("control_b")
        per_target[str(target)] = entry

    violations = []
    for (target, rnd), g in df.groupby(["target", "round"]):
        docked = sorted(int(x) for x in g["n_docked"].unique())
        if len(docked) > 1:
            violations.append({"target": str(target), "round": int(rnd), "n_docked": docked})

    return {"per_target": per_target, "parity_ok": len(violations) == 0,
            "parity_violations": violations}


@click.command()
@click.option("--summary", default="data/dock/sp_f/loop_summary.csv")
def main(summary):
    df = pd.read_csv(summary)
    out = compare_arms(df)
    for target, e in out["per_target"].items():
        print(f"\n=== {target} ===")
        for arm in ("treatment", "control_a", "control_b"):
            if arm in e:
                print(f"  {arm:10} final top10 {e[arm]['final_top10_mean']:.3f}  "
                      f"best {e[arm]['best_overall']:.3f}")
        for k in ("delta_treatment_minus_control_a", "delta_control_a_minus_control_b",
                  "delta_treatment_minus_control_b"):
            if k in e:
                print(f"  {k}: {e[k]:+.3f}")
    print(f"\nbudget parity ok: {out['parity_ok']}")
    if not out["parity_ok"]:
        print(f"  VIOLATIONS: {out['parity_violations']}")


if __name__ == "__main__":
    main()
