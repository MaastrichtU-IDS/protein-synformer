"""Generate the 'notrain' Table III baseline.

The baseline is the pretrained SynFormer-ED decoder+heads with randomly
reinitialized cross-attention and an untrained protein encoder -- i.e. no
protein-conditioning training. Samples molecules per test protein and saves an
`infos` pickle compatible with scripts/reproduce_similarity.py.

Usage:
    python scripts/sample_notrain.py --n-proteins 40 --repeat 40 \
        --out data/evaluations/notrain/infos.pkl
"""
import os
import pickle

import click
import torch
from omegaconf import OmegaConf

from scripts.sample_helpers import load_protein_embeddings, sample
from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.models.synformer import Synformer


def build_notrain_model(config, pretrained_path: str, device: str) -> Synformer:
    model = Synformer(config.model).to(device)
    ckpt = torch.load(pretrained_path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # keep decoder + heads (strip the Lightning "model." prefix); drop encoder
    keep = ("model.decoder.", "model.fingerprint_head.", "model.token_head.", "model.reaction_head.")
    filtered = {k[len("model."):]: v for k, v in state.items() if k.startswith(keep)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"loaded {len(filtered)} decoder/head tensors | missing={len(missing)} unexpected={len(unexpected)}")
    # randomly reinitialize cross-attention (no protein conditioning learned)
    reinit = 0
    for name, module in model.named_modules():
        if name.endswith("multihead_attn") and isinstance(module, torch.nn.MultiheadAttention):
            module._reset_parameters()
            reinit += 1
    print(f"reinitialized {reinit} cross-attention modules; protein encoder left random")
    model.eval()
    return model


@click.command()
@click.option("--config", "config_path", default="configs/prot2drug.yml")
@click.option("--pretrained", default="data/trained_weights/sf_ed_default.ckpt")
@click.option("--n-proteins", default=40, type=int)
@click.option("--repeat", default=40, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--out", "out_path", default="data/evaluations/notrain/infos.pkl")
def main(config_path, pretrained, n_proteins, repeat, seed, out_path):
    torch.manual_seed(seed)
    config = OmegaConf.load(config_path)
    device = "cpu"
    model = build_notrain_model(config, pretrained, device)
    fpindex: FingerprintIndex = pickle.load(open(config.chem.fpindex, "rb"))
    rxn_matrix: ReactantReactionMatrix = pickle.load(open(config.chem.rxn_matrix, "rb"))
    emb = load_protein_embeddings(config.chem.protein_embedding_path)

    targets = list(emb.keys())[:n_proteins]
    infos = {}
    for i, t in enumerate(targets, 1):
        info, _ = sample(t, model, fpindex, rxn_matrix, emb, device, repeat=repeat)
        infos[t] = info
        print(f"[{i}/{len(targets)}] {t}: {len(info)}/{repeat} valid")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pickle.dump(infos, open(out_path, "wb"))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
