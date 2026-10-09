import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark import baseline_forecasts, load_data


class BaselineTests(unittest.TestCase):
    def test_last_and_seasonal(self):
        history = np.arange(1, 13, dtype=float)
        actual = baseline_forecasts(history, 1)
        self.assertEqual(actual["last_value"], 12)
        self.assertEqual(actual["seasonal_naive"], 1)
        self.assertEqual(baseline_forecasts(history, 12)["seasonal_naive"], 12)

    def test_constant(self):
        for h in [1, 3, 6, 12]:
            for yhat in baseline_forecasts(np.full(18, 100.0), h).values():
                self.assertAlmostEqual(yhat, 100)

    def test_future_is_not_used(self):
        series = np.arange(24, dtype=float) + 1
        before = baseline_forecasts(series[:12], 12)
        series[12:] = 999999
        self.assertEqual(before, baseline_forecasts(series[:12], 12))

    def test_missing_and_nonnegative(self):
        history = np.array([50, 40, np.nan, 20, 10, np.nan])
        self.assertTrue(all(np.isfinite(v) and v >= 0
                            for v in baseline_forecasts(history, 12).values()))

    def test_yearly_scale_uses_only_observed_overlap(self):
        history = np.r_[np.arange(1, 13), 2*np.arange(1, 7)]
        self.assertEqual(baseline_forecasts(history, 1)["seasonal_scaled"], 14)

    def test_empty_history_rejected(self):
        with self.assertRaises(ValueError):
            baseline_forecasts([np.nan], 1)

    def test_source_field_mapping_and_missing_month(self):
        frame = pd.DataFrame({"date": ["2023-01", "2023-03"], "territory_id": [1, 1],
                              "category": ["test", "test"], "value": [10, 30]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"source.parquet"
            frame.to_parquet(path)
            _, panel, dates, source_field = load_data(path)
        self.assertEqual(source_field, "value")
        self.assertEqual(len(dates), 3)
        self.assertTrue(np.isnan(panel.iloc[0, 1]))

    def test_duplicate_keys_rejected(self):
        frame = pd.DataFrame({"date": ["2023-01"]*2, "territory_id": [1]*2,
                              "category": ["test"]*2, "value": [10, 20]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"source.parquet"
            frame.to_parquet(path)
            with self.assertRaises(ValueError):
                load_data(path)


if __name__ == "__main__":
    unittest.main()
