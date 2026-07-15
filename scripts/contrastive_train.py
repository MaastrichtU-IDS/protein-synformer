"""Contrastive paralog-discrimination — short fine-tune.

Margin loss pushing LL(route | measured-binder isoform pocket) above LL(route | measured-non-binder
sibling pocket), on within-family train triples (MAPK/CDK/PRKC). Route/pocket LL reuses
ll_target_specificity's pathway featurization + build_pocket_feat + get_log_likelihood_shortcut.

    .venv-train/bin/python -m scripts.contrastive_train --ckpt <SP-C> --out data/ckpt/contrastive_pilot.ckpt
"""
import json

import click
import numpy as np
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

    train = json.load(open("data/dock/contrastive/train_triples.json"))
    held = json.load(open("data/dock/contrastive/heldout_triples.json"))
    g2t = json.load(open("data/dock/contrastive/gene2tid.json"))

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

    opt = T.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        losses = []
        for rf, pb, pn in tr:
            llb = route_pocket_ll(model, rf, pb, device)
            lln = route_pocket_ll(model, rf, pn, device)
            loss = contrastive_loss(llb.unsqueeze(0), lln.unsqueeze(0), margin)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # monitors: train + heldout margin (bind - nonbind), no grad
        model.eval()
        with T.no_grad():
            tr_m = np.mean([(route_pocket_ll(model, rf, pb, device) - route_pocket_ll(model, rf, pn, device)).item()
                            for rf, pb, pn in tr])
            hv_m = np.mean([(route_pocket_ll(model, rf, pb, device) - route_pocket_ll(model, rf, pn, device)).item()
                            for rf, pb, pn in hv]) if hv else float("nan")
        print(f"epoch {ep}: loss={np.mean(losses):.4f} train_margin={tr_m:+.3f} heldout_margin={hv_m:+.3f}",
              flush=True)

    base = T.load(ckpt, map_location="cpu")
    T.save(build_out_checkpoint(base["hyper_parameters"], model.state_dict()), out_ckpt)
    print(f"WROTE {out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
