"""Contrastive gate readout: held-out-family (CSNK1) paralog discrimination win-rate, base SP-C vs the
contrastively-fine-tuned model. Win-rate = fraction of held-out within-family (binder, non-binder) triples
where LL(route|binder pocket) > LL(route|non-binder pocket). Molecule-clustered bootstrap CI.

    .venv-train/bin/python -m scripts.discrim_eval --ft data/ckpt/contrastive_pilot.ckpt
"""
import json
import os

import click
import numpy as np
DATA_DIR = os.environ.get('CONTRASTIVE_DIR','data/dock/contrastive')

from scripts.contrastive_train import SP_C, _route_fields, route_pocket_ll


def winrate(pairs):
    """pairs: list of (ll_bind, ll_nonbind). Fraction where ll_bind > ll_nonbind."""
    return float(np.mean([b > n for b, n in pairs])) if pairs else float("nan")


def _clustered_ci(mol_pairs, nb=5000, seed=42):
    """mol_pairs: dict smiles -> list of (ll_bind, ll_nonbind). Bootstrap resampling MOLECULES."""
    rng = np.random.default_rng(seed)
    mols = list(mol_pairs)
    rs = []
    for _ in range(nb):
        samp = rng.choice(mols, len(mols), replace=True)
        pool = [p for m in samp for p in mol_pairs[m]]
        rs.append(winrate(pool))
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


@click.command()
@click.option("--base", default=SP_C)
@click.option("--ft", default="data/ckpt/contrastive_pilot.ckpt")
def main(base, ft):
    import torch
    from scripts.sample_helpers import load_model
    from synformer.data.pocket_io import load_pockets
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pockets = load_pockets("data/pockets")
    pathways = torch.load("data/synthetic_pathways/filtered_pathways_370000.pth", map_location="cpu")
    canon2raw = {}
    for s in pathways:
        m = Chem.MolFromSmiles(s)
        if m:
            canon2raw[Chem.MolToSmiles(m)] = s
    held = json.load(open(f"{DATA_DIR}/heldout_triples.json"))
    g2t = json.load(open(f"{DATA_DIR}/gene2tid.json"))

    def eval_model(ckpt):
        model, fpindex, _ = load_model(ckpt, None, device)
        fp = fpindex._fp; fp_dim = fp.shape[1]
        model.eval()
        mol_pairs = {}
        with torch.no_grad():
            for smi, bg, ng, fam in held:
                raw = canon2raw.get(smi); bt, nt = g2t.get(bg), g2t.get(ng)
                if raw is None or bt not in pockets or nt not in pockets:
                    continue
                rf = _route_fields(pathways[raw], fp, fp_dim, device)
                llb = route_pocket_ll(model, rf, pockets[bt], device).item()
                lln = route_pocket_ll(model, rf, pockets[nt], device).item()
                mol_pairs.setdefault(smi, []).append((llb, lln))
        flat = [p for ps in mol_pairs.values() for p in ps]
        return mol_pairs, flat

    bmp, bflat = eval_model(base)
    fmp, fflat = eval_model(ft)
    bw, fw = winrate(bflat), winrate(fflat)
    flo, fhi = _clustered_ci(fmp)
    print(f"held-out CSNK1: n_triples={len(fflat)} (mols {len(fmp)})")
    print(f"  base SP-C   win-rate = {bw:.3f}")
    print(f"  contrastive win-rate = {fw:.3f}  clustered CI[{flo:.3f},{fhi:.3f}]  (chance 0.5)")
    print(f"  Δ(FT-base) = {fw-bw:+.3f}")
    verdict = "PASS" if (flo > 0.5 and fw > bw) else "FAIL/inconclusive"
    print(f"  GATE: {verdict}")
    json.dump({"n": len(fflat), "base_winrate": bw, "ft_winrate": fw, "ci": [flo, fhi],
               "delta": fw - bw, "verdict": verdict},
              open(f"{DATA_DIR}/gate_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
