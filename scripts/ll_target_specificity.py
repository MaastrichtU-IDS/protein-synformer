"""Firmed-up target-specificity: LL-based protein retrieval.

For each ligand L (with true protein P), score LL(pathway(L) | P) and LL(pathway(L) | K
random decoy proteins). Molecule fixed => pure protein-conditioning effect. Report:
  - top-1 accuracy: fraction where true P has the highest LL of the K+1 (random = 1/(K+1))
  - mean rank of true P out of K+1 (1=best; random = (K+2)/2)
  - pairwise win-rate: LL(true) > LL(decoy) over all N*K comparisons (random = 50%)
Prints each model's result as soon as it's computed (resilient to being cut off).
"""
import glob, pickle
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
from synformer.models.synformer import Synformer
from synformer.data.common import TokenType
from scripts.sample_helpers import load_protein_embeddings

DEVICE, N, K, SEED = "mps", 60, 10, 42
rng = np.random.default_rng(SEED)

emb = load_protein_embeddings("data/protein_embeddings/embeddings_selection_float16_4973.pth")
pathways = torch.load("data/synthetic_pathways/filtered_pathways_370000.pth", map_location="cpu")
gt = pd.read_csv("/Users/micheldumontier/code/prot2drug/data/papyrus/papyrus_selection_182129.csv") \
        .groupby("target_id")["SMILES"].apply(lambda s: list(dict.fromkeys(s))).to_dict()
fp = pickle.load(open("data/processed/comp_2048/fpindex.pkl", "rb"))._fp
fp_dim = fp.shape[1]

pool = [t for t in gt if t in emb]
pairs = []
for t in pool:
    for s in gt[t]:
        if s in pathways:
            pairs.append((t, s)); break
    if len(pairs) >= N:
        break
print(f"ligands={len(pairs)} decoys_per_ligand={K} pool={len(pool)}", flush=True)

def build_batch(pid, smiles):
    pw = torch.as_tensor(np.asarray(pathways[smiles]), dtype=torch.long)
    tt = pw[:, 0]
    z = torch.zeros_like(pw[:, 1])
    rxn = torch.where(tt == TokenType.REACTION, pw[:, 1], z)
    rct = torch.where(tt == TokenType.REACTANT, pw[:, 1], z)
    rfp = torch.stack([torch.as_tensor(fp[i], dtype=torch.float32) if int(i) != 0 else torch.zeros(fp_dim) for i in rct])
    return {"protein_embeddings": emb[pid].float().unsqueeze(0).to(DEVICE),
            "token_types": tt.unsqueeze(0).to(DEVICE), "rxn_indices": rxn.unsqueeze(0).to(DEVICE),
            "reactant_fps": rfp.unsqueeze(0).to(DEVICE),
            "token_padding_mask": torch.zeros(1, tt.numel(), dtype=torch.bool, device=DEVICE)}

@torch.no_grad()
def ll(model, pid, smiles):
    return float(model.get_log_likelihood_shortcut(build_batch(pid, smiles))["total"].sum().item())

def load_model_only(path):
    ck = torch.load(path, map_location="cpu")
    m = Synformer(OmegaConf.create(ck["hyper_parameters"]["config"]).model).to(DEVICE)
    m.load_state_dict({k[6:]: v for k, v in ck["state_dict"].items()}); m.eval()
    return m

# fixed decoys per ligand (same across models for fairness)
decoys = [list(rng.choice([p for p in pool if p != t], size=K, replace=False)) for t, _ in pairs]

MODELS = {"study_last4": "data/trained_weights/big_pretrained_last4.ckpt",
          "sp2_masked": sorted(glob.glob("logs_gate/sp2_masked/**/checkpoints/last.ckpt", recursive=True))[-1]}

for name, path in MODELS.items():
    model = load_model_only(path)
    ranks, top1, wins, comps = [], 0, 0, 0
    for (t, s), dc in zip(pairs, decoys):
        true_ll = ll(model, t, s)
        dec_ll = [ll(model, d, s) for d in dc]
        rank = 1 + sum(dl > true_ll for dl in dec_ll)   # 1 = true is best
        ranks.append(rank); top1 += (rank == 1)
        wins += sum(true_ll > dl for dl in dec_ll); comps += len(dec_ll)
    ranks = np.array(ranks)
    print(f"\n=== {name} (n={len(ranks)}, K={K}) ===", flush=True)
    print(f"  top-1 accuracy      = {top1/len(ranks)*100:.1f}%   (random {100/(K+1):.1f}%)", flush=True)
    print(f"  mean rank of true   = {ranks.mean():.2f} / {K+1}   (random {(K+2)/2:.1f})", flush=True)
    print(f"  pairwise win-rate   = {wins/comps*100:.1f}%   (random 50%)   [n={comps}]", flush=True)
