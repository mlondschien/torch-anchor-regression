"""Train a model with the CVaR anchor loss and select it by validation risk."""

import copy

import numpy as np
import torch

from .metrics import regression_metrics


@torch.no_grad()
def predict(model, loader, device="cpu"):
    """Concatenated targets, predictions and groups of a loader of (*inputs, y, groups)."""
    model.eval()
    batches = [
        (y, model(*(x.to(device) for x in inputs)).cpu(), groups) for *inputs, y, groups in loader
    ]
    model.train()
    return [torch.cat(tensors).numpy() for tensors in zip(*batches)]


def fit(
    model,
    train_loader,
    valid_loader,
    loss,
    *,
    learning_rate=0.001,
    weight_decay=0.0001,
    warmup_epochs=5,
    epochs=25,
    device="cpu",
):
    """MSE warmup, then training with ``loss``, keeping the epoch with best validation CVaR.

    Loaders yield (*inputs, y, groups) batches with integer groups; ``model(*inputs)``
    predicts y. ``loss(y, yhat, groups)`` is e.g. an ``AnchorLoss``; its parameters, e.g.
    the CVaR threshold, are trained too. Loads the selected weights into ``model`` and
    returns the per-epoch history and the selected epoch.
    """
    model.to(device)
    loss.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "weight_decay": weight_decay},
            {"params": loss.parameters(), "weight_decay": 0},  # not the threshold
        ],
        lr=learning_rate,
    )

    history, best = [], None
    for epoch in range(1, warmup_epochs + epochs + 1):
        losses = []
        for *inputs, y, groups in train_loader:
            y = y.to(device)
            yhat = model(*(x.to(device) for x in inputs))

            if epoch <= warmup_epochs:
                batch_loss = (y - yhat).square().mean()
            else:
                batch_loss = loss(y, yhat, groups.to(device))

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            losses.append(batch_loss.item())

        y, yhat, groups = predict(model, valid_loader, device)
        scores = regression_metrics(y - yhat, groups)
        history.append(dict(epoch=epoch, train_loss=np.mean(losses), **scores))
        print(", ".join(f"{key}={value:.5g}" for key, value in history[-1].items()), flush=True)
        # Select the CVaR-phase checkpoint by validation CVaR_0.2 of the group MSEs.
        if epoch > warmup_epochs and (best is None or scores["cvar_02"] < best["score"]):
            best = dict(
                score=scores["cvar_02"], epoch=epoch, state=copy.deepcopy(model.state_dict())
            )

    model.load_state_dict(best["state"])
    return history, best["epoch"]
