"""Stratified sampling of discrete environments."""

import math

import numpy as np
import torch


class StandardScaler:
    """scikit-learn style StandardScaler for torch tensors."""

    def fit(self, x):
        x = torch.as_tensor(x)
        # All but last axis for multi-dimensional arrays, or all axes for 1d arrays.
        axes = tuple(range(max(x.ndim - 1, 1)))
        self.mean = x.mean(axes)
        scale = x.std(axes, correction=0)
        self.scale = torch.where(scale < 1e-6, 1.0, scale)
        return self

    def transform(self, x):
        return ((torch.as_tensor(x) - self.mean) / self.scale).float()  # cast to float32

    def inverse_transform(self, x):
        return (torch.as_tensor(x) * self.scale + self.mean).numpy()


class StratifiedSampler:
    """Draw stratified batches.

    Draws len(self) batches of size batch_size, each containing environments_per_batch
    distinct environments. Each observation is sampled from once per epoch (up to last
    fractional batch, this is dropped). Thus, each observation has ~same probability to
    be in a single batch. So we don't need to reweight losses by environment size.
    If environments are reasonably uniformly sized, each environment contributes
    ~batch_size / environments_per_batch observations to a batch. If an environment is
    very large (> n / environments_per_batch), it will contribute more observations to
    each batch. The other envs will then contribute fewer.
    """

    def __init__(
        self,
        groups,
        *,
        environments_per_batch=32,
        batch_size=1024,
        min_observations_per_environment=2,
        seed=0,
    ):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.n = len(groups)

        groups = np.ravel(groups)  # in case of (n, 1).
        _, self.codes, self.counts = np.unique(groups, return_inverse=True, return_counts=True)

        # indices[e] holds the indices of the observations in environment e.
        self.indices = np.split(np.argsort(self.codes, kind="stable"), self.counts.cumsum()[:-1])

        self.environments_per_batch = min(environments_per_batch, len(self.counts))

        # Compute `c` s.t. sum_e min(1, c * n_e) = self.environments_per_batch.
        # `p_e = min(1, c * n_e)` is the probability of environment e being in a batch.
        # If an env. has many observations, it will always be in a batch (p_e = 1).
        c = self.environments_per_batch / self.n  # first guess
        while np.minimum(1, c * self.counts).sum() < self.environments_per_batch - 1e-9:
            capped = c * self.counts >= 1
            c = (self.environments_per_batch - capped.sum()) / self.counts[~capped].sum()

        self.probabilities = np.minimum(1, c * self.counts)
        # sizes[e] is number of obs. from env. e in batch. If c * n_e <= 1 for all e,
        # i.e., c = self.environments_per_batch / n, then
        # sizes[e] = batch_size / environments_per_batch for all e.
        # If e is so large that c * n_e > 1, then sizes[e] = batch_size · n_e / n.
        self.sizes = batch_size / self.n * np.maximum(1 / c, self.counts)
        if math.floor(self.sizes.min()) < min_observations_per_environment:
            raise ValueError(
                f"Smallest environments get {math.floor(self.sizes.min())} observations "
                f"per batch, fewer than {min_observations_per_environment=}. "
                "Increase batch_size or lower environments_per_batch."
            )

    def __len__(self):
        return max(1, self.n // self.batch_size)  # Drop the last fractional batch

    def __iter__(self):
        num_batches, k = len(self), self.environments_per_batch

        # Environment e is in repeats[e] ~ self.probabilities[e] * num_batches batches.
        order = self.rng.permutation(len(self.counts))
        cumulative = self.rng.uniform() + (self.probabilities[order] * num_batches).cumsum()
        repeats = np.diff(np.floor(cumulative), prepend=0).astype(int)

        # environments_per_batch[b, :] is the list of environments in the b-th batch.
        environments_per_batch = np.repeat(order, repeats).reshape(k, num_batches).T
        # queues[e] is a shuffled list of indices in environment e. queue[e][:sizes[e]]
        # will be indices of observations from e in first batch where e appears. And
        # so on.
        queues = [self.rng.permutation(indices) for indices in self.indices]

        # positions[e] is how many observations of env. e have been drawn so far.
        # In the next batch where e appears, draw indices
        # queues[e][positions[e]:positions[e]+sizes[e]]
        positions = np.zeros(len(self.counts), dtype=int)
        for environments in environments_per_batch[self.rng.permutation(num_batches)]:
            # get integer values sizes that sum to batch_size.
            sizes = np.diff(np.round(self.sizes[environments].cumsum()), prepend=0).astype(int)

            batch = []
            # draw with mode=wrap for special case where environments has too few
            # observations
            for e, size in zip(environments, sizes):
                draw = np.arange(positions[e], positions[e] + size)
                batch.append(np.take(queues[e], draw, mode="wrap"))
                positions[e] += size
            yield np.concatenate(batch)
