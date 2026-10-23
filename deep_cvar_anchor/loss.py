"""CVaR anchor loss with unit observation weights and discrete environments."""

import torch
from torch import nn
from torch.nn import functional as F


class AnchorLoss(nn.Module):
    """CVaR anchor loss for environments (groups) as anchor.

    The CVaR uses the Rockafellar-Uryasev form with a learned threshold and a softplus
    hinge of temperature ``tau`` (``tau=0`` is the exact hinge).
    ``debias`` corrects the squared means.
    """

    def __init__(self, alpha=0.1, gamma=1.0, tau=0.001, debias=True):
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha, self.gamma, self.tau, self.debias = alpha, gamma, tau, debias
        self.threshold = nn.Parameter(torch.tensor(0.0))

    def forward(self, y, yhat, groups):
        _, groups, counts = torch.unique(groups, return_inverse=True, return_counts=True)
        residuals = y - yhat

        # Construct group-means and group-variances (shape: (num_groups,)).
        # new_zeros -> same dtype and device. index_add_ is in-place weighted bincount.
        means = residuals.new_zeros(len(counts)).index_add_(0, groups, residuals) / counts
        # These variances are still unnormalized, normalized below.
        variances = residuals.new_zeros(len(counts))
        variances.index_add_(0, groups, (residuals - means[groups]).square())

        if self.debias:
            # E[ (1 / n \sum_i x_i)^2 ] = E[ x_i ]^2 + Var[ x_i ] / n
            # We use debiased variances below for the (M_A (Y - f(x)))^2 term as well,
            # s.t. gamma=1, alpha=1 recovers MSE
            variances = variances / (counts - 1).clamp_min(1)
            scores = means.square() - variances / counts
        else:
            variances = variances / counts
            scores = means.square()

        weights = counts.to(residuals.dtype) / len(residuals)  # relative group weights

        if self.alpha == 1:
            penalty = (weights * scores).sum()  # weighted sum of squared proj residuals

        else:
            # hinge / RU version of CVaR: min_t t + 1/alpha E[ (score - t)_+ ]
            excess = scores - self.threshold
            hinge = self.tau * F.softplus(excess / self.tau) if self.tau else excess.relu()
            penalty = self.threshold + (weights * hinge).sum() / self.alpha

        return (weights * variances).sum() + self.gamma * penalty
