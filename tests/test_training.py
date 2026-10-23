import json

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from scripts.train_NYC_taxi import main, predict_from_checkpoint
from torch_anchor_regression import regression_metrics


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


def test_train_select_and_reload(tmp_path):
    rng = np.random.default_rng(0)
    data = {}
    for split in ("train", "valid", "test"):
        x = rng.normal(size=(30, 2))
        data.update(
            {
                f"x_{split}": x,
                f"y_{split}": 20 + x[:, 0],
                f"group_{split}": np.repeat([3, 7, 9], [1, 9, 20]),
            }
        )
    np.savez(tmp_path / "data.npz", **data)
    arguments = f"--data-path {tmp_path / 'data.npz'} --output {tmp_path / 'run'} --width 8"
    result = CliRunner().invoke(main, arguments.split())
    assert result.exit_code == 0, result.output + str(result.exception)

    # Selected by validation CVaR among the CVaR-phase epochs.
    history = pd.read_csv(tmp_path / "run/history.csv")
    scores = json.loads((tmp_path / "run/scores.json").read_text())
    assert scores["best_epoch"] == history.epoch[history.cvar_02[5:].idxmin()]

    # The checkpoint reproduces the test scores in the original target units.
    prediction = predict_from_checkpoint(tmp_path / "run/model.pt", data["x_test"])
    expected = regression_metrics(data["y_test"] - prediction, data["group_test"])
    assert scores["mse"] == pytest.approx(expected["mse"])
