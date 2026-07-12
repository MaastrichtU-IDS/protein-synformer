import torch
from synformer.molopt.dpo import dpo_loss


def test_dpo_loss_lower_when_policy_prefers_winner():
    # reference indifferent; policy raises winner, lowers loser -> loss should drop
    llref_w = torch.tensor([0.0]); llref_l = torch.tensor([0.0])
    good = dpo_loss(torch.tensor([2.0]), torch.tensor([-2.0]), llref_w, llref_l, beta=0.5)
    bad  = dpo_loss(torch.tensor([-2.0]), torch.tensor([2.0]), llref_w, llref_l, beta=0.5)
    assert good.item() < bad.item()


def test_dpo_loss_reference_cancels():
    # equal policy margins but shifted reference -> loss depends on (policy - reference) margin
    l1 = dpo_loss(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0)
    l2 = dpo_loss(torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]), beta=1.0)
    assert abs(l1.item() - l2.item()) < 1e-5   # same (policy-ref) margins -> same loss


def test_dpo_loss_positive_and_finite():
    v = dpo_loss(torch.tensor([0.5, -0.5]), torch.tensor([0.0, 0.0]),
                 torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), beta=0.1)
    assert torch.isfinite(v) and v.item() > 0
