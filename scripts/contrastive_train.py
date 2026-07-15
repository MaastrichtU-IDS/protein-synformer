"""Contrastive paralog-discrimination — short fine-tune.

Margin loss pushing LL(route | measured-binder isoform pocket) above LL(route | measured-non-binder
sibling pocket), on within-family train triples (MAPK/CDK/PRKC). Route/pocket LL reuses
ll_target_specificity's pathway featurization + build_pocket_feat + get_log_likelihood_shortcut.

    .venv-train/bin/python -m scripts.contrastive_train --ckpt <SP-C> --out data/ckpt/contrastive_pilot.ckpt
"""
import json
import os

import click
import numpy as np
DATA_DIR = os.environ.get('CONTRASTIVE_DIR','data/dock/contrastive')
import torch
import torch.nn.functional as F

from synformer.data.common import TokenType

SP_C = "logs/pocket/2607091019-32f2194@powered-specificity/2026_07_09__10_19_15/checkpoints/epoch=1-step=2255.ckpt"


def contrastive_loss(ll_bind, ll_nonbind, margin=2.0):
    return F.softplus(margin - (ll_bind - ll_nonbind)).mean()


def _route_fields(pathway, fp, fp_dim, device):
    pw = torch.as_tensor(np.asarray(pathway), dtype=torch.long)
    tt = pw[:, 0]
    z = torch.zeros_like(pw[:, 1])
    rxn = torch.where(tt == TokenType.REACTION, pw[:, 1], z)
    rct = torch.where(tt == TokenType.REACTANT, pw[:, 1], z)
    rfp = torch.stack([torch.as_tensor(fp[int(i)], dtype=torch.float32) if int(i) != 0
                       else torch.zeros(fp_dim) for i in rct])
    return {"token_types": tt.unsqueeze(0).to(device), "rxn_indices": rxn.unsqueeze(0).to(device),
            "reactant_fps": rfp.unsqueeze(0).to(device),
            "token_padding_mask": torch.zeros(1, tt.numel(), dtype=torch.bool, device=device)}


def route_pocket_ll(model, route_fields, pocket, device):
    from scripts.sample_helpers import build_pocket_feat
    batch = {**build_pocket_feat(pocket, 1, device), **route_fields}
    return model.get_log_likelihood_shortcut(batch)["total"].sum()


@click.command()
@click.option("--ckpt", default=SP_C)
@click.option("--out", "out_ckpt", default="data/ckpt/contrastive_pilot.ckpt")
@click.option("--lr", default=1e-5, type=float)
@click.option("--epochs", default=3, type=int)
@click.option("--margin", default=2.0, type=float)
def main(ckpt, out_ckpt, lr, epochs, margin):
    import torch as T
    from scripts.sample_helpers import load_model
    from synformer.data.pocket_io import load_pockets
    from scripts.dpo_train import build_out_checkpoint

    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    model, fpindex, _ = load_model(ckpt, None, device)
    fp = fpindex._fp
    fp_dim = fp.shape[1]
    pockets = load_pockets("data/pockets")
    pathways = T.load("data/synthetic_pathways/filtered_pathways_370000.pth", map_location="cpu")
    # canonical-smiles keyed pathway lookup
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    canon2raw = {}
    for s in pathways:
        m = Chem.MolFromSmiles(s)
        if m:
            canon2raw[Chem.MolToSmiles(m)] = s

    train = json.load(open(f"{DATA_DIR}/train_triples.json"))
    held = json.load(open(f"{DATA_DIR}/heldout_triples.json"))
    g2t = json.load(open(f"{DATA_DIR}/gene2tid.json"))

    def prep(triples):
        out = []
        for smi, bg, ng, fam in triples:
            raw = canon2raw.get(smi)
            btid, ntid = g2t.get(bg), g2t.get(ng)
            if raw is None or btid not in pockets or ntid not in pockets:
                continue
            out.append((_route_fields(pathways[raw], fp, fp_dim, device), pockets[btid], pockets[ntid]))
        return out

    tr = prep(train)
    hv = prep(held)
    print(f"usable train triples={len(tr)} heldout={len(hv)}", flush=True)

    def norm_ll(rf, pocket):
        # length-normalize so routes of different lengths contribute comparably (margin scale stable);
        # per-route normalization leaves the bind-vs-nonbind SIGN (win-rate) unchanged.
        n = rf["token_types"].shape[1]
        return route_pocket_ll(model, rf, pocket, device) / n

    opt = T.optim.AdamW(model.parameters(), lr=lr)
    base = T.load(ckpt, map_location="cpu")
    best_state, best_trm = None, -1e9
    for ep in range(epochs):
        # FULL-BATCH: accumulate over all train triples, one step per epoch (batch-1 SGD on 80 triples
        # thrashed; this stabilizes optimization).
        model.train()
        opt.zero_grad()
        llb = T.stack([norm_ll(rf, pb) for rf, pb, pn in tr])
        lln = T.stack([norm_ll(rf, pn) for rf, pb, pn in tr])
        loss = contrastive_loss(llb, lln, margin)
        loss.backward(); opt.step()
        model.eval()
        with T.no_grad():
            trm = float((T.stack([norm_ll(rf, pb) for rf, pb, pn in tr]) -
                         T.stack([norm_ll(rf, pn) for rf, pb, pn in tr])).mean())
            hvm = (float((T.stack([norm_ll(rf, pb) for rf, pb, pn in hv]) -
                          T.stack([norm_ll(rf, pn) for rf, pb, pn in hv])).mean()) if hv else float("nan"))
        print(f"epoch {ep}: loss={loss.item():.4f} train_margin(norm)={trm:+.4f} heldout_margin(norm)={hvm:+.4f}",
              flush=True)
        if trm > best_trm:   # save BEST-train-margin state, not the final (advisor)
            best_trm = trm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    T.save(build_out_checkpoint(base["hyper_parameters"], best_state), out_ckpt)
    print(f"WROTE {out_ckpt} (best train_margin(norm)={best_trm:+.4f})", flush=True)


if __name__ == "__main__":
    main()
