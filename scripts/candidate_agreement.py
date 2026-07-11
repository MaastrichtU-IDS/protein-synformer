"""Candidate-regime scorer-agreement analysis: per-target smina<->Boltz Spearman (candidate vs
known/random regime), smina-top hacking percentile, and selection overlap. Headline = regime contrast."""
from __future__ import annotations

import click
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _load_candidate_boltz(csv) -> pd.DataFrame:
    return (pd.read_csv(csv).rename(columns={"smiles": "molecule", "affinity_pred": "boltz"})
            [["target", "molecule", "boltz"]].dropna(subset=["boltz"]))


def spearman_by_target(frame) -> dict:
    out = {}
    for target, g in frame.groupby("target"):
        if len(g) < 3:
            out[str(target)] = float("nan"); continue
        rho = spearmanr(-g.smina.to_numpy(float), -g.boltz.to_numpy(float)).correlation
        out[str(target)] = float(rho)
    return out


def _pct_rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def hacking_percentile(frame, k: int = 5) -> dict:
    out = {}
    for target, g in frame.groupby("target"):
        g = g.copy()
        g["boltz_pct"] = _pct_rank(-g.boltz.to_numpy(float))        # 1 = strongest boltz
        top = g.sort_values("smina").head(k)                        # k strongest by smina
        out[str(target)] = float(top["boltz_pct"].mean())
    return out


def _topk(g, col, k):
    return set(g.sort_values(col).head(k).molecule) if col != "rankmean" else \
        set(g.assign(rankmean=(pd.Series(-g.smina.to_numpy(float)).rank(ascending=False).to_numpy()
                               + pd.Series(-g.boltz.to_numpy(float)).rank(ascending=False).to_numpy()))
            .sort_values("rankmean").head(k).molecule)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def selection_overlap(frame, k: int = 5) -> dict:
    out = {}
    for target, g in frame.groupby("target"):
        s = set(g.sort_values("smina").head(k).molecule)
        b = set(g.sort_values("boltz").head(k).molecule)
        c = _topk(g, "rankmean", k)
        out[str(target)] = {"smina_vs_boltz": _jaccard(s, b),
                            "consensus_vs_smina": _jaccard(c, s),
                            "consensus_vs_boltz": _jaccard(c, b)}
    return out


def regime_contrast(candidate_frame, knownrandom_frame) -> dict:
    cand = spearman_by_target(candidate_frame)
    kr = spearman_by_target(knownrandom_frame)
    per = {t: {"candidate": cand.get(t, float("nan")), "known_random": kr.get(t, float("nan"))}
           for t in set(cand) | set(kr)}
    finite = lambda d: [v for v in d.values() if v == v]
    return {"per_target": per,
            "mean_candidate": float(np.mean(finite(cand))) if finite(cand) else float("nan"),
            "mean_known_random": float(np.mean(finite(kr))) if finite(kr) else float("nan")}


@click.command()
@click.option("--candidate-boltz", default="data/dock/sp_cc_candidate_boltz.csv")
@click.option("--pocket-scores", default="data/dock/dock_scores_pocket.csv")
@click.option("--kr-boltz", default="data/dock/sp_cs_boltz_controls.csv")
@click.option("--dock-scores", default="data/dock/dock_scores.csv")
@click.option("--targets", default="O43570_WT,P06537_WT,P10721_WT,P02753_WT,P0C559_WT")
def main(candidate_boltz, pocket_scores, kr_boltz, dock_scores, targets):
    from scripts.candidate_boltz import load_candidates
    from scripts.consensus_score import load_smina, load_boltz, build_frame

    tlist = [t.strip() for t in targets.split(",")]
    cand = load_candidates(pocket_scores, tlist).merge(_load_candidate_boltz(candidate_boltz),
                                                       on=["target", "molecule"], how="inner")
    kr = build_frame(load_smina(dock_scores, tlist), load_boltz(kr_boltz))  # known/random regime
    rc = regime_contrast(cand, kr)
    print("REGIME CONTRAST — smina<->Boltz Spearman (higher = agree):")
    for t, e in sorted(rc["per_target"].items()):
        print(f"  {t:12} candidate {e['candidate']:+.3f}   known/random {e['known_random']:+.3f}")
    print(f"  MEAN         candidate {rc['mean_candidate']:+.3f}   known/random {rc['mean_known_random']:+.3f}")
    print("\nHACKING — mean Boltz percentile of smina-top-5 (low = smina-top are Boltz-weak):")
    for t, p in sorted(hacking_percentile(cand).items()):
        print(f"  {t:12} {p:.2f}")
    print("\nSELECTION OVERLAP (Jaccard, top-5):")
    for t, o in sorted(selection_overlap(cand).items()):
        print(f"  {t:12} smina~boltz {o['smina_vs_boltz']:.2f}  cons~smina {o['consensus_vs_smina']:.2f}  cons~boltz {o['consensus_vs_boltz']:.2f}")


if __name__ == "__main__":
    main()
