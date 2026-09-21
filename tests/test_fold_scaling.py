import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from meit_model.cv import DevelopmentData
from meit_model.fold_scaling import FoldScalerSource, checkpoint_data, rescale_data
from meit_model.risk_loss import risk_aware_masked_mae


class FoldScalingTests(unittest.TestCase):
    def test_unique_observed_inputs_only_and_no_validation_influence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, segment, starts in (("train", 1, [0, 1]), ("validation", 2, [0])):
                np.savez(root / f"{split}.npz", segment_index=np.full(len(starts), segment),
                         input_start_bin_index=starts)
            (root / "scaler.json").write_text("{}")
            bins = pd.DataFrame({"segment_index": np.repeat([1, 2], 46),
                                 "observed": True,
                                 "observed_depth_m": np.arange(92, dtype=float) / 100})
            bins.loc[3, "observed"] = False
            path = root / "minute_bins.csv.gz"
            bins.to_csv(path, index=False)
            first = FoldScalerSource(root).fit([0, 1])
            expected = np.delete(np.arange(31) / 100, 3)
            self.assertEqual(first["unique_observed_bin_count"], 30)
            self.assertAlmostEqual(first["mean_m"], expected.mean())
            self.assertAlmostEqual(first["std_m"], expected.std())
            # Training targets outside input coverage and all validation bins.
            bins.loc[31:, "observed_depth_m"] = 999
            bins.to_csv(path, index=False)
            self.assertEqual(first, FoldScalerSource(root).fit([0, 1]))

    def test_transform_preserves_metres_masks_and_original_arrays(self):
        source = {"mean_m": .1, "std_m": .2}
        target = {"mean_m": .2, "std_m": .05}
        data = DevelopmentData(np.ones((1, 30, 1), np.float32),
                               np.full((1, 15, 1), np.nan, np.float32),
                               np.zeros((1, 15, 1), bool), np.array([0]), np.array([0]), 1)
        converted = rescale_data(data, source, target)
        np.testing.assert_allclose(converted.X * .05 + .2, data.X * .2 + .1)
        np.testing.assert_array_equal(data.X, 1)
        self.assertTrue(np.isnan(converted.Y).all())
        self.assertEqual(converted.X.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "Retrain"):
            checkpoint_data(data, source, {}, {}, [0])


class RiskLossTests(unittest.TestCase):
    def test_penalty_does_not_cancel_and_masked_nan_has_zero_gradient(self):
        prediction = torch.tensor([.2, .2, .9], requires_grad=True)
        target = torch.tensor([.4, .4, float("nan")])
        mask = torch.tensor([True, True, False])
        scaler = {"mean_m": 0., "std_m": 1.}
        base = risk_aware_masked_mae(prediction, target, mask, scaler, underprediction_weight=1.)
        penalized = risk_aware_masked_mae(prediction, target, mask, scaler, underprediction_weight=2.)
        self.assertAlmostEqual(penalized.item(), 2 * base.item())
        penalized.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([-1., -1., 0.]))

    def test_overprediction_and_low_water_are_not_penalized(self):
        loss = risk_aware_masked_mae(torch.tensor([.5, .1]), torch.tensor([.4, .2]),
                                     torch.tensor([True, True]), {"mean_m": 0., "std_m": 1.})
        self.assertAlmostEqual(loss.item(), .1, places=6)


if __name__ == "__main__":
    unittest.main()
