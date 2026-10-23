import itertools

import pytest
import torch

from deep_cvar_anchor.loss import AnchorLoss


@pytest.mark.parametrize("debias", [False, True])
def test_alpha_and_gamma_one_is_mse(debias):
    y = torch.tensor([2.0, -1.0, 3.0, 0.5, 1.0, -2.0])
    yhat = torch.zeros_like(y, requires_grad=True)
    groups = torch.tensor([0, 1, 1, 2, 2, 2])
    actual = AnchorLoss(alpha=1, gamma=1, debias=debias)(y, yhat, groups)
    expected = (y - yhat).square().mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, yhat)[0], torch.autograd.grad(expected, yhat)[0]
    )


def test_cvar_with_fractional_tail():
    # Squared group means 9, 4, 1 and variances 1. The worst half of observations is
    # all of the first group and half of the second: CVaR = (9 + 4 / 2) / 1.5 = 22 / 3,
    # attained at threshold 4: 4 + (9 - 4) / 3 / 0.5. Loss = 1 + 2 * 22 / 3.
    y = torch.tensor([2.0, 4.0, 1.0, 3.0, 0.0, 2.0])
    loss = AnchorLoss(alpha=0.5, gamma=2, tau=0, debias=False)
    with torch.no_grad():
        loss.threshold.fill_(4)
    groups = torch.arange(3).repeat_interleave(2)
    torch.testing.assert_close(loss(y, torch.zeros_like(y), groups), torch.tensor(47 / 3))


def test_debiased_pairs_average_to_full_data():
    # Averaged over all pairs drawn without replacement, the debiased loss equals the
    # debiased loss of the whole group.
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    pairs = torch.tensor(list(itertools.combinations(data.tolist(), 2))).double().flatten()
    loss = AnchorLoss(alpha=1, gamma=2)
    actual = loss(pairs, torch.zeros_like(pairs), torch.arange(6).repeat_interleave(2))
    expected = loss(data, torch.zeros_like(data), torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(actual, expected)
