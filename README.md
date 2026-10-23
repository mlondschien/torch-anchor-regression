# torch-anchor-regression

A small PyTorch project for regression with discrete environments. The package holds
the stratified sampler, the CVaR anchor loss, metrics and a generic training loop.
`scripts/train_NYC_taxi.py` trains an MLP on NYC taxi trips, using pickup zones as
environments; `scripts/train_vitaldb.py` a waveform CNN on VitalDB patients.

```sh
python -m pip install -e .
python scripts/train_NYC_taxi.py --output results/taxi_cvar
python scripts/train_NYC_taxi.py --alpha 1 --gamma 1 --output results/taxi_erm
python scripts/train_NYC_taxi.py --alpha 1 --gamma 2 --output results/taxi_anchor
python scripts/train_NYC_taxi.py --alpha 0.2 --gamma 2 --environments-per-batch 32 --batch-size 2048 --output results/taxi_cvar_2048
```

Defaults: three hidden layers of 64 ReLU units, AdamW at `0.001`, network weight
decay `0.0001`, **5 MSE warmup epochs followed by 25 CVaR epochs**, debiasing,
`alpha=0.1`, `gamma=1`, and smoothing temperature `tau=0.001`.
Use `--device mps` for Apple GPUs. `python scripts/train_NYC_taxi.py --help` lists the options.

## Sampling and loss

```python
from torch_anchor_regression.data import StratifiedSampler

sampler = StratifiedSampler(groups, environments_per_batch=32, batch_size=1024, seed=0)
for indices in sampler:
    # One minibatch; iterate over sampler again for the next epoch.
    ...
```

Every batch contains exactly `environments_per_batch` distinct environments and has
exactly `batch_size` observations. Environment `e` is included with
probability `min(1, c * n_e)`, where `c` makes these probabilities sum to
`environments_per_batch`; systematic sampling then picks exactly that many distinct
environments. Environments with probability one are in every batch and contribute
their share `batch_size * n_e / n` of observations; the others contribute
`batch_size / (c * n)` each. **In expectation, each environment contributes in
proportion to its size, so every observation has unit weight.**

Each epoch plans all its batches at once, so that environment `e` is in about
`p_e * len(sampler)` of them, and draws observations from a shuffled queue per
environment. An epoch has `len(sampler) = n // batch_size` batches, dropping the last
fractional one, and uses (almost) every observation exactly once: about 98.5% on taxi. On the taxi data with the
defaults, 10 zones are in every batch with about 31 to 44 trips each and the other 22
zones contribute about 30 trips each. These sizes are fractional; each batch rounds
them up or down so that they sum to exactly `batch_size`.

`criterion(y, yhat, groups)` takes three flat tensors. It weights each environment's
statistics by its share of the batch:

```text
score_e = mean_residual_e² - sample_variance_e / count_e
weight_e = count_e / batch_size
loss = sum_e weight_e * sample_variance_e
       + gamma * (threshold + sum_e weight_e * hinge(score_e - threshold) / alpha)
hinge(x) = tau * softplus(x / tau)
```

The sample variance uses denominator `count_e - 1`. A batch with a single
observation of an environment has no variance estimate: its variance contribution
is zero and its squared residual is left uncorrected.
`--no-debias` uses denominator `count_e` and omits the squared-mean correction.
Negative corrected scores are retained. `tau=0` uses the exact hinge. At `alpha=1`,
the penalty is simply the mean score; `alpha=gamma=1` gives batch MSE. Debiasing
targets population squared means under iid sampling within environments; it does
not remove bias introduced by the nonlinear hinge.

The RU threshold is learned jointly with the network, without weight decay, starting
at zero. The loss is in standardized target units. The feature and target scalers
are stored in `model.pt` alongside the pickled model.

## Data and evaluation

The prepared `data/taxi.npz` uses July 2019 for training/validation and April 2020
for testing. The target is log total fare; the 26 features include distance,
duration, passenger count, pickup time, and categorical indicators. To prepare it
again from the local monthly Parquet files:

```sh
python -m pip install -e '.[datasets]'
python -m datasets.prepare_taxi
```

The preparer defaults to a 300,000-trip July sample. Its `--n-trips` option changes
that cap. It reserves approximately 10% of July trips within each zone for validation.

For another tabular dataset, supply an NPZ with `x_train` of shape
`(observations, features)`, vector targets `y_train` and environment labels
`group_train`, and the same keys for `valid` and `test`.
Training standardizes features and targets; predictions and evaluation metrics are
in the original target units.
Or pass these arrays in a dictionary to `train(data, output, ...)` in
`scripts/train_NYC_taxi.py`. For other data or models, build loaders of
`(*inputs, y, groups)` batches and call
`torch_anchor_regression.fit(model, train_loader, valid_loader, AnchorLoss(alpha, gamma), ...)`
directly; it calls `model(*inputs)`, runs the warmup and CVaR epochs and loads the
selected weights into the model. `scripts/train_vitaldb.py` does this for a waveform
CNN with demographics:

```sh
python -m datasets.prepare_vitaldb
python scripts/train_vitaldb.py --output results/vitaldb_cvar
```

Validation selects the CVaR-phase checkpoint with the lowest **trip-weighted CVaR of
the environment MSEs at tail fraction 0.2** (`cvar_02`). The selected
checkpoint is then evaluated on the test split. `mse`, `rmse` and `mae` pool all
observations. `q_08`, `q_09` and `q_095` are quantiles and `cvar_02`, `cvar_01` and
`cvar_005` CVaRs of the environment MSEs, with environments weighted by their number of
observations. Taxi MSE is in squared log-fare units.
Warmup checkpoints and test outcomes do not select the model.

Each output directory must be empty. Runs save the model and preprocessing in
`model.pt`, along with `config.json`, `history.csv`, `scores.json` and per-environment
test MSEs in `test_environments.csv`. Compute predictions with:

```python
from scripts.train_NYC_taxi import predict_from_checkpoint

prediction = predict_from_checkpoint("results/taxi_cvar/model.pt", raw_features)
```

## Code

| File | Purpose |
| --- | --- |
| `scripts/train_NYC_taxi.py` | NYC taxi data, model, loaders, outputs and command line |
| `scripts/train_vitaldb.py` | VitalDB waveform CNN with demographics; patients as environments |
| `torch_anchor_regression/data.py` | Stratified environment sampler |
| `torch_anchor_regression/loss.py` | CVaR anchor loss with unit observation weights |
| `torch_anchor_regression/metrics.py` | Pooled errors and quantiles/CVaRs of environment MSEs |
| `torch_anchor_regression/train.py` | Generic training loop, validation selection and prediction |
| `datasets/prepare_taxi.py` | Taxi filtering, features, and splits |

The broader training interface, including probit and chi-square, lives in
[`experiments/train.py`](experiments/train.py); see its
[documentation](experiments/multidataset.md). Existing dataset-specific experiments
are documented in [`experiments/README.md`](experiments/README.md).

```sh
python -m pytest tests experiments/tests
pre-commit run --all-files  # black, ruff and file checks
```
