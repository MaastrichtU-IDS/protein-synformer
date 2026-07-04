"""Bounded validation of the real training path on MPS.

Exercises exactly what train.py does -- ProjectionDataModule (real data) +
SynformerWrapper + a PyTorch-Lightning Trainer -- but with fast_dev_run=True
(1 train batch + 1 val batch, no checkpoints, no full loop). Confirms the
datamodule produces valid batches from the real data and that a forward +
backward + optimizer step runs on MPS before committing to a multi-hour run.

Usage:
    python -m scripts.validate_train --device mps --batch 4
"""
import click
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from synformer.data.projection_dataset_new import ProjectionDataModule
from synformer.models.wrapper import SynformerWrapper

PAIRS = "/Users/micheldumontier/code/prot2drug/data/papyrus/papyrus_selection_182129.csv"


@click.command()
@click.option("--config", "config_path", default="configs/prot2drug.yml")
@click.option("--device", default="mps")
@click.option("--batch", default=4, type=int)
@click.option("--resume", default=None, type=click.Path(exists=True),
              help="Optional pretrained ckpt to warm-start decoder/heads (the real FT path).")
def main(config_path, device, batch, resume):
    cfg = OmegaConf.load(config_path)
    cfg.system.device = device
    # wire to data we actually have (the study's exact split files are absent)
    cfg.chem.protein_molecule_pairs_train_path = PAIRS
    cfg.chem.protein_molecule_pairs_val_path = PAIRS
    cfg.model.decoder.lora = False
    cfg.model.decoder.lora_rank = 0

    datamodule = ProjectionDataModule(cfg, batch_size=batch, num_workers=0, **cfg.data)
    model = SynformerWrapper(config=cfg, args={"config_path": config_path, "resume": resume})

    if resume:
        ckpt = torch.load(resume, map_location="cpu")
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        keep = ("model.decoder.", "model.fingerprint_head.", "model.token_head.", "model.reaction_head.")
        filtered = {k: v for k, v in state.items() if k.startswith(keep)}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"warm-started {len(filtered)} tensors | missing={len(missing)} unexpected={len(unexpected)}")

    trainer = pl.Trainer(
        accelerator=device, devices=1, fast_dev_run=True,
        logger=False, enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=datamodule)
    print("\nVALIDATE_TRAIN: OK - real datamodule + SynformerWrapper + MPS train/val step ran")


if __name__ == "__main__":
    main()
