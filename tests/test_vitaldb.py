import numpy as np
import pandas as pd

from datasets.prepare_vitaldb import make_windows


def test_windows():
    # 24 s at 500 Hz: three 8 s windows. The second contains a missing sample and is
    # dropped. Labels are averaged within each window.
    time = np.arange(12000) / 500
    pressure = 80 + 15 * np.sin(2 * np.pi * time)
    pressure[4001] = np.nan
    sparse_time = np.full(len(time), np.nan)  # VitalDB stores only a few timestamps
    sparse_time[[0, 1, -1]] = time[[0, 1, -1]]
    waveform = pd.DataFrame({"Time": sparse_time, "SNUADC/ART": pressure})
    labels = pd.DataFrame({"Time": [0, 6, 8, 10, 16, 20], "Vigileo/SV": [40, 60, 70, 90, 100, 120]})
    x, y, starts = make_windows(waveform, labels)
    assert x.shape == (2, 1000, 1)
    np.testing.assert_array_equal(y, [50, 110])
    np.testing.assert_array_equal(starts, [0, 16])
    np.testing.assert_allclose(x[0, :, 0], pressure[:4000].reshape(1000, 4).mean(1), atol=1e-5)
