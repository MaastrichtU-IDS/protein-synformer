"""Consensus-scorer discrimination benchmark: does consensus(smina, Boltz) separate
known binders from random decoys more robustly (worst-case AUROC) than either alone?"""
from __future__ import annotations

import click
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_smina(dock_scores_csv, targets) -> pd.DataFrame:
    d = pd.read_csv(dock_scores_csv)
    d = d[(d.target.isin(targets)) & (d.pocket == d.target) & (d.source.isin(["known", "random"]))]
    out = d[["target", "molecule", "source", "score"]].copy()
    out["is_known"] = out.source == "known"
    out = out.rename(columns={"score": "smina"}).drop(columns=["source"])
    return out.dropna(subset=["smina"]).drop_duplicates(["target", "molecule"])


def load_boltz(boltz_csv) -> pd.DataFrame:
    d = pd.read_csv(boltz_csv).rename(columns={"smiles": "molecule", "affinity_pred": "boltz"})
    return d[["target", "molecule", "boltz"]].dropna(subset=["boltz"]).drop_duplicates(["target", "molecule"])


def build_frame(smina_df, boltz_df) -> pd.DataFrame:
    return smina_df.merge(boltz_df, on=["target", "molecule"], how="inner")


def _auroc(y_known, strength) -> float:
    return float(roc_auc_score(y_known.astype(int), strength))


def benchmark(frame, min_known: int = 5) -> dict:
    per_target, skipped = {}, []
    for target, g in frame.groupby("target"):
        n_known = int(g.is_known.sum())
        n_rand = int((~g.is_known).sum())
        if n_known < min_known or n_rand < 2:
            skipped.append(str(target))
            continue
        s_smina = -g.smina.to_numpy(dtype=float)   # strength = -score
        s_boltz = -g.boltz.to_numpy(dtype=float)
        rankmean = (pd.Series(s_smina).rank().to_numpy() + pd.Series(s_boltz).rank().to_numpy()) / 2.0
        zsum = _z(s_smina) + _z(s_boltz)
        y = g.is_known
        per_target[str(target)] = {
            "smina": _auroc(y, s_smina), "boltz": _auroc(y, s_boltz),
            "rankmean": _auroc(y, rankmean), "zsum": _auroc(y, zsum),
            "n_known": n_known, "n_random": n_rand,
        }
    scorers = ["smina", "boltz", "rankmean", "zsum"]
    mean = {s: float(np.mean([t[s] for t in per_target.values()])) if per_target else float("nan") for s in scorers}
    worst = {s: float(np.min([t[s] for t in per_target.values()])) if per_target else float("nan") for s in scorers}
    return {"per_target": per_target, "mean": mean, "worst": worst, "skipped": skipped}


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


@click.command()
@click.option("--dock-scores", default="data/dock/dock_scores.csv")
@click.option("--boltz", "boltz_csv", default="data/dock/sp_cs_boltz_controls.csv")
@click.option("--targets", default="O43570_WT,P10721_WT,P02753_WT,P0C559_WT")
@click.option("--min-known", default=5, type=int)
def main(dock_scores, boltz_csv, targets, min_known):
    tlist = [t.strip() for t in targets.split(",")]
    frame = build_frame(load_smina(dock_scores, tlist), load_boltz(boltz_csv))
    out = benchmark(frame, min_known=min_known)
    print("per-target AUROC (known vs random):")
    for t, e in out["per_target"].items():
        print(f"  {t:12} smina {e['smina']:.3f}  boltz {e['boltz']:.3f}  "
              f"rankmean {e['rankmean']:.3f}  zsum {e['zsum']:.3f}  (k={e['n_known']} r={e['n_random']})")
    print(f"skipped: {out['skipped']}")
    for agg in ("mean", "worst"):
        o = out[agg]
        print(f"{agg:6} smina {o['smina']:.3f}  boltz {o['boltz']:.3f}  "
              f"rankmean {o['rankmean']:.3f}  zsum {o['zsum']:.3f}")
    w = out["worst"]
    best_single = max(w["smina"], w["boltz"])
    print(f"\nWORST-CASE rescue: rankmean {w['rankmean']:.3f} vs best-single {best_single:.3f} "
          f"-> {'consensus more robust' if w['rankmean'] > best_single else 'no rescue'}")


if __name__ == "__main__":
    main()
