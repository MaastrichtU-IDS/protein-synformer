"""Artifact control for the contrastive gate: base held-out win-rate was 0.263 (should be ~0.5 if the eval
is unstructured), so the FT's 0.684 may be flipping a route/pocket artifact, not learning selectivity.

Compares, on held-out CSNK1, for base + FT:
  - TRUE win-rate: route vs its own (binder, non-binder) isoform pockets.
  - SCRAMBLE win-rate: the binder/non-binder isoform-pocket-pair is PERMUTED across drugs (route↔its-own
    measured pockets link broken), averaged over P permutations. If TRUE≈SCRAMBLE, the metric doesn't
    depend on the true binding labels => artifact. If TRUE high & SCRAMBLE≈0.5 => real within-sample.
Also dumps per-isoform mean LL (fixed-pocket bias check).

    .venv-train/bin/python -m scripts.discrim_control --ft data/ckpt/contrastive_pilot.ckpt
"""
import json
import os
import click
import numpy as np
DATA_DIR = os.environ.get('CONTRASTIVE_DIR','data/dock/contrastive')
from scripts.contrastive_train import SP_C, _route_fields, route_pocket_ll


@click.command()
@click.option("--base", default=SP_C)
@click.option("--ft", default="data/ckpt/contrastive_pilot.ckpt")
@click.option("--perms", default=200, type=int)
def main(base, ft, perms):
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
    rng = np.random.default_rng(42)

    def eval_model(ckpt):
        model, fpindex, _ = load_model(ckpt, None, device)
        fp = fpindex._fp; fp_dim = fp.shape[1]
        model.eval()
        rows = []  # (route_fields, binder_tid, nonbinder_tid, binder_gene, nonbinder_gene)
        for smi, bg, ng, fam in held:
            raw = canon2raw.get(smi); bt, nt = g2t.get(bg), g2t.get(ng)
            if raw is None or bt not in pockets or nt not in pockets:
                continue
            rows.append((_route_fields(pathways[raw], fp, fp_dim, device), bt, nt, bg, ng))
        # LL of each route under every distinct held-out isoform pocket (for scramble + per-isoform bias)
        tids = sorted({r[1] for r in rows} | {r[2] for r in rows})
        with torch.no_grad():
            ll = {}  # (route_idx, tid) -> LL
            for i, r in enumerate(rows):
                for t in tids:
                    ll[(i, t)] = route_pocket_ll(model, r[0], pockets[t], device).item()
        true_w = np.mean([ll[(i, r[1])] > ll[(i, r[2])] for i, r in enumerate(rows)])
        # scramble: permute the (binder,nonbinder) tid pairs across routes
        pairs = [(r[1], r[2]) for r in rows]
        sw = []
        for _ in range(perms):
            perm = rng.permutation(len(pairs))
            sw.append(np.mean([ll[(i, pairs[perm[i]][0])] > ll[(i, pairs[perm[i]][1])] for i in range(len(rows))]))
        # per-isoform mean LL (fixed pocket preference?)
        iso = {t: np.mean([ll[(i, t)] for i in range(len(rows))]) for t in tids}
        return true_w, np.mean(sw), np.std(sw), iso, len(rows)

    for name, ck in [("base", base), ("FT", ft)]:
        tw, sw, ss, iso, n = eval_model(ck)
        print(f"{name}: TRUE win-rate={tw:.3f}  SCRAMBLE win-rate={sw:.3f}±{ss:.3f}  (n={n})", flush=True)
        print(f"   per-isoform mean route-LL: {json.dumps({k.replace('_WT',''): round(v,1) for k,v in iso.items()})}", flush=True)
    print("CONTROL DONE", flush=True)


if __name__ == "__main__":
    main()
