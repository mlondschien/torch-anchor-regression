"""Regression metrics over discrete environments."""

import numpy as np


def regression_metrics(residuals, groups):
    """Return pooled mse, rmse and mae; quantiles and CVaRs of the group MSEs."""
    _, codes, counts = np.unique(groups, return_inverse=True, return_counts=True)
    risks = np.bincount(codes, weights=residuals**2) / counts
    mse = float(np.mean(residuals**2))
    scores = dict(mse=mse, rmse=mse**0.5, mae=float(np.mean(np.abs(residuals))))

    for level in [0.8, 0.9, 0.95]:
        quantile = np.quantile(risks, level, weights=counts, method="inverted_cdf")
        scores[f"q_{level:g}".replace(".", "")] = float(quantile)

    order = np.argsort(-risks)
    weights = counts[order] / counts.sum()
    for alpha in [0.2, 0.1, 0.05]:
        tail = np.clip(alpha - (weights.cumsum() - weights), 0, weights)
        scores[f"cvar_{alpha:g}".replace(".", "")] = float(tail @ risks[order] / alpha)

    return scores
