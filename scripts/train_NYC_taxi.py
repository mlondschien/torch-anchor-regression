"""Train an MLP on NYC taxi trips; pickup zones are the environments."""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, TensorDataset

from torch_anchor_regression import AnchorLoss, fit, predict, regression_metrics
from torch_anchor_regression.data import StandardScaler, StratifiedSampler


class MLP(nn.Module):
    """Simple ReLU network for tabular inputs of shape (batch, features)."""

    def __init__(self, n_features, width=64, depth=3):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(n_features, width), nn.ReLU()])
            n_features = width
        self.layers = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def train(
    data,
    output,
    *,
    width=64,
    depth=3,
    alpha=0.1,
    gamma=1.0,
    tau=0.001,
    debias=True,
    environments_per_batch=32,
    batch_size=1024,
    seed=0,
    device="cpu",
    **options,
):
    """Standardize, train with ``fit`` and score once on the test split.

    ``data`` holds x, y and group arrays for the train, valid and test splits. Features
    are tabular or fixed-length sequences; targets and group labels are vectors.
    ``options`` are passed to ``fit``.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    x, y = data["x_train"], data["y_train"]
    model = MLP(x.shape[-1], width, depth)

    x_scaler = StandardScaler().fit(x)
    y_scaler = StandardScaler().fit(y)

    def loader(split, sampler):
        # The samplers yield index arrays; batch_size=None indexes the dataset with them.
        return DataLoader(
            TensorDataset(
                x_scaler.transform(data[f"x_{split}"]),
                y_scaler.transform(data[f"y_{split}"]),
                torch.as_tensor(data[f"group_{split}"]),
            ),
            sampler=sampler,
            batch_size=None,
        )

    def sequential(split):
        return BatchSampler(range(len(data[f"y_{split}"])), 65536, drop_last=False)

    sampler = StratifiedSampler(
        data["group_train"],
        environments_per_batch=environments_per_batch,
        min_observations_per_environment=environments_per_batch // 4,
        batch_size=batch_size,
        seed=seed,
    )
    history, best_epoch = fit(
        model,
        loader("train", sampler),
        loader("valid", sequential("valid")),
        AnchorLoss(alpha, gamma, tau, debias),
        device=device,
        **options,
    )

    model.cpu()
    checkpoint = dict(model=model, x_scaler=x_scaler, y_scaler=y_scaler, best_epoch=best_epoch)
    torch.save(checkpoint, output / "model.pt")

    _, prediction, _ = predict(model, loader("test", sequential("test")))
    residuals = data["y_test"] - y_scaler.inverse_transform(prediction)
    test_scores = regression_metrics(residuals, data["group_test"])
    scores = dict(
        test_scores,
        validation_score=history[best_epoch - 1]["cvar_02"],
        best_epoch=best_epoch,
        batch_size=sampler.batch_size,
        steps_per_epoch=len(sampler),
    )
    (output / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    table = pd.DataFrame(dict(environment=data["group_test"], residual=residuals))
    table.assign(squared=residuals**2).groupby("environment").agg(
        count=("residual", "size"), mean_residual=("residual", "mean"), mse=("squared", "mean")
    ).to_csv(output / "test_environments.csv")
    return scores


def predict_from_checkpoint(path, X):
    """Predict from raw features with a model saved by ``train``, in target units."""
    saved = torch.load(path, weights_only=False)
    saved["model"].eval()
    with torch.no_grad():
        prediction = saved["model"](saved["x_scaler"].transform(X))
    return saved["y_scaler"].inverse_transform(prediction)


@click.command(context_settings={"show_default": True})
@click.option("--data-path", type=click.Path(exists=True, path_type=Path), default="data/taxi.npz")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--alpha", type=click.FloatRange(0, 1, min_open=True), default=0.1)
@click.option("--gamma", type=click.FloatRange(0), default=1.0)
@click.option("--tau", type=click.FloatRange(0), default=0.001)
@click.option("--debias/--no-debias", default=True)
@click.option("--width", type=click.IntRange(1), default=64)
@click.option("--depth", type=click.IntRange(1), default=3)
@click.option("--learning-rate", type=click.FloatRange(0, min_open=True), default=0.001)
@click.option("--weight-decay", type=click.FloatRange(0), default=0.0001)
@click.option("--environments-per-batch", type=click.IntRange(1), default=32)
@click.option("--batch-size", type=click.IntRange(1), default=1024)
@click.option("--warmup-epochs", type=click.IntRange(0), default=5)
@click.option("--epochs", type=click.IntRange(1), default=25, help="CVaR epochs after warmup.")
@click.option("--seed", type=click.IntRange(0), default=0)
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default="cpu")
def main(data_path, output, **options):
    """Fit with unit observation weights and select the checkpoint by validation risk."""
    with np.load(data_path, allow_pickle=False) as archive:
        scores = train(dict(archive), output, **options)
    config = dict(options, data_path=str(data_path.resolve()))
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    click.echo(f"Test MSE={scores['mse']:.5f}, CVaR_0.2={scores['cvar_02']:.5f}; {output}")


if __name__ == "__main__":
    main()
