import unittest

import numpy as np
import pandas as pd

from meit_data.io import Event
from meit_data.preprocess import (bin_observations, complete_bin_ends, event_sampling_weights,
                                  event_segments, fit_scaler, make_windows, normalize_windows,
                                  scaler_bin_mask, validate_config)
from test_io import config


class CausalPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.config = config()

    def bins(self, seconds, values=None, ends=None):
        seconds = np.asarray(seconds, dtype=float)
        if values is None:
            values = seconds / 6000
        if ends is None:
            ends = complete_bin_ends(seconds[0], seconds[-1])
        bins = bin_observations(seconds, np.asarray(values), np.asarray(ends), self.config["quality"])
        bins["bin_end_unix_seconds"] = bins.bin_end_relative_seconds + 1704067200
        return bins

    def test_exact_45_minutes_produces_one_window(self):
        bins = self.bins(np.arange(46) * 60)
        windows, stats = make_windows(bins, self.config)
        self.assertEqual(windows["X"].shape, (1, 30, 1))
        self.assertEqual(windows["Y"].shape, (1, 15, 1))
        self.assertEqual(stats["accepted_windows"], 1)
        self.assertAlmostEqual(windows["X"][0, -1, 0], .3)
        self.assertAlmostEqual(windows["Y"][0, 0, 0], .31)
        self.assertAlmostEqual(windows["Y"][0, -1, 0], .45)

    def test_right_boundary_and_last_observation(self):
        bins = self.bins([0, 20, 59, 60, 61, 120], [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(bins.observed_depth_m, [3, 5])
        partial = self.bins([0, 60, 61, 121], [0, 3, 4, 5], [60, 120])
        self.assertEqual(partial.observed_depth_m.tolist(), [3, 4])

    def test_future_values_cannot_change_input(self):
        seconds = np.arange(61) * 60
        first = self.bins(seconds)
        depths = seconds / 6000
        depths[seconds > 1800] = 999
        second = self.bins(seconds, depths)
        pd.testing.assert_frame_equal(first.iloc[:30], second.iloc[:30])
        prefix = self.bins(seconds[seconds <= 1800])
        pd.testing.assert_frame_equal(first.iloc[:30], prefix)
        first_windows, _ = make_windows(first, self.config)
        second_windows, _ = make_windows(second, self.config)
        np.testing.assert_array_equal(first_windows["X"][0], second_windows["X"][0])

    def test_forward_fill_does_not_fabricate_target(self):
        seconds = np.delete(np.arange(46) * 60, 35)
        bins = self.bins(seconds)
        self.assertFalse(bins.observed.iloc[34])
        self.assertTrue(np.isfinite(bins.causal_input_depth_m.iloc[34]))
        self.assertTrue(np.isnan(bins.observed_depth_m.iloc[34]))
        windows, stats = make_windows(bins, self.config)
        self.assertEqual(stats["accepted_windows"], 1)
        self.assertEqual(windows["Y_mask"].sum(), 14)
        self.assertTrue(np.isnan(windows["Y"][0, 4, 0]))
        normalized = normalize_windows(windows, {"mean_m": .1, "std_m": .2})
        self.assertTrue(np.isnan(normalized["Y"][0, 4, 0]))

    def test_integer_observation_requirements(self):
        for missing, accepted in (([3, 10, 20], 1), ([3, 10, 20, 25], 0), ([35], 1), ([35, 40], 0)):
            with self.subTest(missing=missing):
                windows, _ = make_windows(self.bins(np.delete(np.arange(46) * 60, missing)), self.config)
                self.assertEqual(len(windows["X"]), accepted)

    def test_fill_limits_and_no_initial_zero_padding(self):
        bins = self.bins([0, 60, 600], [1, 2, 9], np.arange(60, 601, 60))
        np.testing.assert_allclose(bins.causal_input_depth_m.iloc[:3], [2, 2, 2])
        self.assertTrue(np.isnan(bins.causal_input_depth_m.iloc[3]))
        limited = dict(self.config["quality"], max_last_observation_age_seconds=100)
        bins = bin_observations(np.array([0., 600.]), np.array([1., 9.]), np.array([60, 120]), limited)
        self.assertEqual(bins.causal_input_depth_m.iloc[0], 1)
        self.assertTrue(np.isnan(bins.causal_input_depth_m.iloc[1]))
        bins = self.bins([120, 180], [1, 2], [60, 120, 180])
        self.assertTrue(np.isnan(bins.causal_input_depth_m.iloc[0]))

    def test_long_gap_splits_without_bridging_and_keeps_event_grid(self):
        start = pd.Timestamp("2024-01-01T00:00:00Z")
        seconds = np.r_[np.arange(31) * 60, 3001 + np.arange(31) * 60]
        event = Event("event", "sensor", start, start + pd.Timedelta(seconds=float(seconds[-1])),
                      2, seconds, seconds / 10000)
        segments = list(event_segments(event, self.config))
        self.assertEqual(len(segments), 2)
        for _, _, bins in segments:
            windows, _ = make_windows(bins, self.config)
            self.assertEqual(len(windows["X"]), 0)
            self.assertTrue((bins.bin_end_relative_seconds % 60 == 0).all())
        self.assertEqual(segments[1][2].bin_end_relative_seconds.iloc[0], 3120)
        np.testing.assert_array_equal(complete_bin_ends(11999, 12150), [12060, 12120])

    def test_scaler_uses_unique_observed_bins_only(self):
        bins = self.bins(np.delete(np.arange(47) * 60, 4))
        windows, _ = make_windows(bins, self.config)
        mask = scaler_bin_mask(bins, windows, 30)
        # Two overlapping histories cover bins 1..31, with minute 4 missing.
        self.assertEqual(mask.sum(), 30)
        self.assertFalse(mask[3])
        self.assertFalse(mask[31:].any())
        scaler = fit_scaler(bins.observed_depth_m.to_numpy()[mask])
        expected = np.delete(np.arange(1, 32), 3) / 100
        self.assertAlmostEqual(scaler["mean_m"], expected.mean())
        self.assertAlmostEqual(scaler["std_m"], expected.std())
        before = dict(scaler)
        changed = dict(windows, Y=windows["Y"] * 1000)
        normalize_windows(changed, scaler)
        self.assertEqual(scaler, before)
        with self.assertRaises(ValueError):
            fit_scaler([1, 1])

    def test_event_weights_have_equal_total_mass(self):
        ids = np.array([0, 0, 0, 1, 2, 2])
        weights = event_sampling_weights(ids)
        for event in (0, 1, 2):
            self.assertAlmostEqual(weights[ids == event].sum(), 1)

    def test_config_rejects_changed_contract(self):
        validate_config(self.config)
        self.config["quality"]["target_fill"] = "linear"
        with self.assertRaises(ValueError):
            validate_config(self.config)


if __name__ == "__main__":
    unittest.main()
