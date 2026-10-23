import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from torch_anchor_regression import AnchorLoss, fit, predict, regression_metrics
from torch_anchor_regression.data import StratifiedSampler


def test_regression_metrics():
    # Group MSEs 5, 3 and 1 with 10, 30 and 60 observations. Weighted by group size, 90%
    # of observations have MSE <= 3; the worst 20% are all of a and 10 of the 30 of b.
    residuals = np.repeat([5**0.5, 3**0.5, 1.0], [10, 30, 60])
    scores = regression_metrics(residuals, np.repeat(["a", "b", "c"], [10, 30, 60]))
    expected = dict(
        mse=2,
        mae=(10 * 5**0.5 + 30 * 3**0.5 + 60) / 100,
        q_08=3,
        q_09=3,
        q_095=5,
        cvar_02=4,
        cvar_01=5,
        cvar_005=5,
    )
    assert {key: scores[key] for key in expected} == pytest.approx(expected)


def test_fit_selects_and_loads_best_epoch():
    torch.manual_seed(0)
    groups = torch.as_tensor(np.repeat([3, 7, 9], [20, 30, 50]))
    x = torch.randn(100, 2)
    y = x[:, 0] + 0.5 * (groups == 3) + 0.1 * torch.randn(100)
    dataset = TensorDataset(x, y, groups)
    sampler = StratifiedSampler(groups.numpy(), environments_per_batch=3, batch_size=50)
    # The sampler yields index arrays; batch_size=None indexes the dataset with them.
    train_loader = DataLoader(dataset, sampler=sampler, batch_size=None)
    valid_loader = DataLoader(dataset, batch_size=40)
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Flatten(0))

    history, best_epoch = fit(
        model, train_loader, valid_loader, AnchorLoss(alpha=0.5, gamma=2), warmup_epochs=2, epochs=3
    )

    # Selected by validation CVaR among the epochs after warmup, and loaded into the model.
    assert len(history) == 5
    assert best_epoch == 3 + int(np.argmin([row["cvar_02"] for row in history[2:]]))
    y, yhat, groups = predict(model, valid_loader)
    assert regression_metrics(y - yhat, groups)["cvar_02"] == pytest.approx(
        history[best_epoch - 1]["cvar_02"]
    )
