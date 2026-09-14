import unittest

import numpy as np

from meit_data.inference_input import prepare_input
from test_io import config


class InferenceInputTests(unittest.TestCase):
    def setUp(self):
        self.config = config()
        self.scaler = {"mean_m": .1, "std_m": .2, "depth_unit": "m"}
        self.origin = 1704067207

    def prepare(self, relative, now, depths=None, received=None):
        relative = np.asarray(relative)
        depths = relative / 6000 if depths is None else depths
        return prepare_input(self.origin + relative, depths, self.origin + now, self.origin,
                             self.config, self.scaler,
                             received_at_unix_seconds=None if received is None else self.origin + np.asarray(received))

    def test_ready_with_only_30_minutes_and_no_future_labels(self):
        result = self.prepare(np.arange(31) * 60, 1800)
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["X"].shape, (1, 30, 1))
        np.testing.assert_allclose(result["X"][0, :, 0], (np.arange(1, 31) / 100 - .1) / .2, atol=1e-7)
        np.testing.assert_array_equal(result["target_bin_end_unix_seconds"], self.origin + 1800 + np.arange(1, 16) * 60)

    def test_short_history_is_not_zero_padded(self):
        result = self.prepare(np.arange(30) * 60, 1740)
        self.assertEqual(result["status"], "WARMUP")
        self.assertIsNone(result["X"])

    def test_no_observations_or_stale_history_invalidates_forecast(self):
        for times, now in (([], 1800), (np.arange(31) * 60, 2040)):
            result = self.prepare(times, now)
            self.assertEqual(result["status"], "INSUFFICIENT_DATA")
            self.assertIsNone(result["X"])

    def test_long_gap_requires_new_history(self):
        relative = np.r_[np.arange(31) * 60, 2400 + np.arange(16) * 60]
        result = self.prepare(relative, 3300)
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(result["X"])

    def test_future_and_late_received_observations_cannot_change_input(self):
        relative = np.arange(46) * 60
        reference = self.prepare(relative[:31], 1800)
        values = relative / 6000
        values[31:] = 999
        actual = self.prepare(relative, 1800, values)
        np.testing.assert_array_equal(reference["X"], actual["X"])
        values[31:] = np.nan
        actual = self.prepare(relative, 1800, values)
        np.testing.assert_array_equal(reference["X"], actual["X"])
        received = relative.copy()
        received[30] = 1810
        delayed = self.prepare(relative, 1800, values, received)
        self.assertEqual(delayed["status"], "READY")
        self.assertFalse(delayed["X_observed"][0, -1, 0])
        self.assertEqual(delayed["X_age_seconds"][0, -1, 0], 60)
        self.assertNotEqual(delayed["X"][0, -1, 0], reference["X"][0, -1, 0])

    def test_session_grid_is_fixed_and_conflicting_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            self.prepare(np.arange(31) * 60, 1801)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.prepare([0, 60, 60, 1800], 1800, [0, .1, .2, .3])


if __name__ == "__main__":
    unittest.main()
