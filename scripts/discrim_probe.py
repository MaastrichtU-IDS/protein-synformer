"""Feasibility gate for training-level targeting: does the CURRENT SP-C pocket model already use its
conditioning to rank a known binder's route as more likely under its TRUE target pocket than under
MISMATCHED pockets? If yes, there is a latent discrimination signal a contrastive objective could sharpen;
if not (fraction ~0.5, mean Δ ~0), the pocket code is being ignored and contrastive training must create
the signal from scratch.

Method: take held-out (target, known-binder-route) val examples; compute per-route log-likelihood under
the TRUE pocket vs pockets rolled across the batch (mismatched, same routes). Report mean(ll_true-ll_mis)
and fraction(ll_true>ll_mis), averaged over several roll shifts.

    .venv-train/bin/python -m scripts.discrim_probe
"""
import numpy as np
import torch
from omegaconf import OmegaConf

CKPT = "logs/pocket/2607091019-32f2194@powered-specificity/2026_07_09__10_19_15/checkpoints/epoch=1-step=2255.ckpt"


def _ll(model, code, mask, b):
    out = model.get_log_likelihood(
        code=code, code_padding_mask=mask,
        token_types=b["token_types"], rxn_indices=b["rxn_indices"],
        reactant_fps=b["reactant_fps"], token_padding_mask=b["token_padding_mask"],
    )
    return out["total"].sum(dim=1)  # [B] per-route total log-likelihood


def main():
    from scripts.sample_helpers import load_model
    from synformer.data.projection_dataset_new import ProjectionDataModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _fp, _rx = load_model(CKPT, None, device)
    model.eval()

    import types
    config = OmegaConf.load("configs/pocket.yml")
    dm = ProjectionDataModule(config, batch_size=128, num_workers=0, **config.data)
    # setup() only reads self.trainer for a presence guard (no sharding here) -> give it a stub
    try:
        dm.trainer = types.SimpleNamespace()
    except Exception:
        dm._trainer = types.SimpleNamespace()
    dm.setup("fit")
    batch = next(iter(dm.val_dataloader()))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    pocket_keys = ["pocket_ca", "pocket_cb", "pocket_restype", "pocket_padding_mask"]
    with torch.inference_mode():
        code_t, mask_t, _ = model.encode(batch)
        ll_true = _ll(model, code_t, mask_t, batch).float().cpu().numpy()

        diffs, fracs = [], []
        for shift in (1, 2, 3, 7, 13):   # several mismatched pairings (roll pockets across batch)
            mis = dict(batch)
            for k in pocket_keys:
                if k in batch:
                    mis[k] = torch.roll(batch[k], shifts=shift, dims=0)
            code_m, mask_m, _ = model.encode(mis)
            ll_mis = _ll(model, code_m, mask_m, batch).float().cpu().numpy()
            d = ll_true - ll_mis
            diffs.append(d.mean())
            fracs.append((d > 0).mean())

    n = len(ll_true)
    print(f"n_routes={n}")
    print(f"mean(ll_true - ll_mismatch) over shifts: {np.mean(diffs):+.3f}  (per-shift {[round(x,2) for x in diffs]})")
    print(f"fraction(ll_true > ll_mismatch): {np.mean(fracs):.3f}  (per-shift {[round(x,2) for x in fracs]})")
    print(f"ll_true mean={ll_true.mean():.2f}")
    # crude significance: is fraction meaningfully above 0.5?
    f = np.mean(fracs)
    se = (0.25 / n) ** 0.5
    print(f"fraction vs 0.5: z~{(f-0.5)/se:.2f} (per shift, n={n})")


if __name__ == "__main__":
    main()
