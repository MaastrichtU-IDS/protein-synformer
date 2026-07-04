"""Benchmark training step time + peak memory for the Big model on a device.

Builds the real Big architecture from a config, runs a few forward+backward+step
iterations on a synthetic batch, and reports mean step time. Used to judge whether
fine-tuning is practical on this machine.

Usage:
    python -m scripts.bench_train --config configs/prot2drug.yml --device mps \
        --batch 16 --protein-len 512 --mode full
    python -m scripts.bench_train ... --mode last4   # freeze all but last 4 decoder layers + heads
"""
import time

import click
import torch
from omegaconf import OmegaConf

from synformer.data.common import TokenType
from synformer.models.synformer import Synformer


def make_batch(b, seq_len, prot_len, fp_dim, device):
    return {
        "protein_embeddings": torch.randn(b, prot_len, 1152, device=device),
        "token_types": torch.randint(0, int(max(TokenType)) + 1, (b, seq_len), device=device),
        "rxn_indices": torch.randint(0, 120, (b, seq_len), device=device),
        "reactant_fps": (torch.rand(b, seq_len, fp_dim, device=device) < 0.03).float(),
        "token_padding_mask": torch.zeros(b, seq_len, dtype=torch.bool, device=device),
    }


@click.command()
@click.option("--config", "config_path", default="configs/prot2drug.yml")
@click.option("--device", default="mps")
@click.option("--batch", default=16, type=int)
@click.option("--protein-len", default=512, type=int)
@click.option("--seq-len", default=24, type=int)
@click.option("--steps", default=8, type=int)
@click.option("--mode", default="full", type=click.Choice(["full", "last4", "last1"]))
def main(config_path, device, batch, protein_len, seq_len, steps, mode):
    cfg = OmegaConf.load(config_path)
    cfg.model.decoder.lora = False
    cfg.model.decoder.lora_rank = 0
    fp_dim = cfg.chem.fp_option.morgan_n_bits
    model = Synformer(cfg.model).to(device)

    if mode in ("last4", "last1"):
        keep = 4 if mode == "last4" else 1
        n_layers = cfg.model.decoder.num_layers
        for name, p in model.named_parameters():
            if name.startswith("decoder.dec.layers."):
                idx = int(name.split(".")[3])
                if idx < n_layers - keep and "multihead_attn" not in name:
                    p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"mode={mode} device={device} batch={batch} protein_len={protein_len}")
    print(f"params: {n_all/1e6:.1f}M total | {n_train/1e6:.1f}M trainable")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    batch_data = make_batch(batch, seq_len, protein_len, fp_dim, device)

    times = []
    for i in range(steps):
        t0 = time.time()
        opt.zero_grad()
        loss_dict, _ = model.get_loss_shortcut(batch_data)
        loss = sum(loss_dict.values())
        loss.backward()
        opt.step()
        if device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        if i > 0:  # drop warmup
            times.append(dt)
        print(f"  step {i}: {dt:.2f}s  loss={float(loss):.3f}")

    mean = sum(times) / len(times)
    print(f"\nmean step time (excl. warmup): {mean:.2f}s")
    for total in (5000, 28000):
        print(f"  ~{total} steps -> {mean*total/3600:.1f} h")
    if device == "mps":
        print(f"peak MPS memory: {torch.mps.driver_allocated_memory()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
