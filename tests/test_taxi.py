import numpy as np
import pandas as pd

from datasets.prepare_taxi import load_month


def test_filters_target_and_features(tmp_path):
    # Only the first trip passes: the others have an outlier fare, no passengers, a too
    # short distance, a missing rate code or a pickup outside the month.
    pickup = pd.to_datetime(["2019-07-01 12:00"] * 5 + ["2019-08-01 12:00"])
    frame = pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=10),
            "passenger_count": [1, 1, 0, 1, 1, 1],
            "trip_distance": [2, 2, 2, 0.1, 2, 2],
            "RatecodeID": [1, 1, 1, 1, np.nan, 1],
            "payment_type": [1] * 6,
            "total_amount": [20, 500, 20, 20, 20, 20],
            "PULocationID": [42] * 6,
        }
    )
    frame.to_parquet(tmp_path / "yellow_tripdata_2019-07.parquet")
    x, y, groups = load_month(tmp_path, "2019-07", 100)
    assert x.shape == (1, 26)
    np.testing.assert_allclose(y, [np.log(20)])
    np.testing.assert_array_equal(groups, [42])
    distance, duration = np.log(2), np.log(10)
    np.testing.assert_allclose(
        x[0, :5], [distance, distance**2, duration, duration**2, distance * duration]
    )
