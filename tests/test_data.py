import numpy as np
import pytest

from torch_anchor_regression.data import StandardScaler, StratifiedSampler


def test_sampler():
    # 40 environments of 50 to 500 observations and a large one in every batch.
    sizes = np.append(np.random.default_rng(0).integers(50, 500, size=40), 20000)
    groups = np.repeat(np.arange(41), sizes)
    sampler = StratifiedSampler(groups, environments_per_batch=8, batch_size=512)
    batches = list(sampler)
    assert len(batches) == len(groups) // 512
    for batch in batches:
        assert len(batch) == 512
        assert len(np.unique(groups[batch])) == 8
        assert 40 in groups[batch]
    draws = np.bincount(np.concatenate(batches), minlength=len(groups))
    assert np.mean(draws == 1) > 0.95


def test_sampler_raises_for_too_few_observations_per_environment():
    with pytest.raises(ValueError, match="Increase batch_size"):
        StratifiedSampler(np.repeat([0, 1], [1, 1000]), environments_per_batch=2, batch_size=4)


def test_standard_scaler():
    x = np.random.default_rng(0).normal(3, 2, size=(50, 4, 2))
    x[..., 1] = 7  # constant feature: centered, not scaled
    scaler = StandardScaler().fit(x)
    np.testing.assert_allclose(scaler.mean, [x[..., 0].mean(), 7])
    np.testing.assert_allclose(scaler.scale, [x[..., 0].std(), 1])
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(x)), x, atol=1e-5)
