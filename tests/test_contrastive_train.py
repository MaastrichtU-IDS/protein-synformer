import torch
from scripts.contrastive_train import contrastive_loss


def test_contrastive_loss_rewards_binder_over_nonbinder():
    good = contrastive_loss(torch.tensor([3.0]), torch.tensor([0.0]), margin=2.0)  # bind >> non
    bad = contrastive_loss(torch.tensor([0.0]), torch.tensor([3.0]), margin=2.0)   # inverted
    assert good.item() < bad.item()
    assert torch.isfinite(good) and good.item() >= 0
