import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import SequentialSampler, WeightedRandomSampler
    from meit_data.torch_data import (ForecastDataset, depth_to_m, horizon_errors,
                                      make_loader, masked_mae)
    from meit_data.pipeline import run_pipeline
    import test_pipeline


@unittest.skipUnless(TORCH_AVAILABLE, "Optional PyTorch dependency not installed")
class TorchHandoffTests(unittest.TestCase):
    def test_masked_labels_do_not_change_loss_or_gradients(self):
        prediction = torch.tensor([1., 2., 3.], requires_grad=True)
        target = torch.tensor([0., float("nan"), 4.])
        mask = torch.tensor([True, False, True])
        first = masked_mae(prediction, target, mask)
        first.backward()
        gradient = prediction.grad.clone()
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(gradient[1], 0)
        prediction.grad.zero_()
        target[1] = 1e10
        second = masked_mae(prediction, target, mask)
        second.backward()
        self.assertEqual(first.item(), second.item())
        torch.testing.assert_close(gradient, prediction.grad)
        with self.assertRaises(ValueError):
            masked_mae(prediction, target, torch.zeros(3, dtype=torch.bool))

    def test_metric_counts_and_sums_ignore_missing_labels(self):
        prediction = torch.tensor([[[1.], [2.]], [[2.], [4.]], [[5.], [2.]]])
        target = torch.tensor([[[0.], [float("nan")]], [[0.], [1.]], [[0.], [2.]]])
        mask = torch.isfinite(target)
        result = horizon_errors(prediction, target, mask)
        torch.testing.assert_close(result["observed_count"], torch.tensor([3, 2]))
        torch.testing.assert_close(result["absolute_error_sum_m"], torch.tensor([8., 3.], dtype=torch.float64))
        left = horizon_errors(prediction[:2], target[:2], mask[:2])
        right = horizon_errors(prediction[2:], target[2:], mask[2:])
        for key in result:
            torch.testing.assert_close(result[key], left[key] + right[key])

    def test_inverse_scaling_clips_in_metres_only(self):
        values = torch.tensor([-2., -.5, 0., 2.])
        scaler = {"mean_m": .1, "std_m": .1}
        torch.testing.assert_close(depth_to_m(values, scaler, clip_negative=True), torch.tensor([0., .05, .1, .3]))

    def test_real_loader_sampling_lstm_and_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, settings = test_pipeline.PipelineIntegrationTests().fixture(root)
            run_pipeline(source, settings, root / "data", root / "report.json")
            train = ForecastDataset(root / "data", "train")
            validation = ForecastDataset(root / "data", "validation")
            risk = ForecastDataset(root / "data", "test_risk")
            loader = make_loader(train, batch_size=8)
            self.assertIsInstance(loader.sampler, WeightedRandomSampler)
            self.assertTrue(loader.sampler.replacement)
            self.assertEqual(loader.sampler.num_samples, len(train))
            self.assertIsInstance(make_loader(validation).sampler, SequentialSampler)
            self.assertIsInstance(make_loader(risk).sampler, SequentialSampler)
            first = next(iter(make_loader(train, batch_size=8, seed=42)))
            second = next(iter(make_loader(train, batch_size=8, seed=42)))
            torch.testing.assert_close(first["X"], second["X"])
            self.assertNotIn("sample_weight", first)
            self.assertEqual(first["X"].dtype, torch.float32)
            self.assertEqual(first["Y_mask"].dtype, torch.bool)
            self.assertEqual(first["target_bin_end_unix_seconds"].shape, (8, 15))
            model = torch.nn.LSTM(1, 64, 1, batch_first=True)
            head = torch.nn.Linear(64, 15)
            hidden, _ = model(first["X"])
            prediction = head(hidden[:, -1]).unsqueeze(-1)
            self.assertEqual(prediction.shape, first["Y"].shape)
            loss = masked_mae(prediction, first["Y"], first["Y_mask"])
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in list(model.parameters()) + list(head.parameters())))
            ids = train.arrays["event_index"].numpy()
            for event in np.unique(ids):
                self.assertAlmostEqual(train.sampling_weights.numpy()[ids == event].sum(), 1)


if __name__ == "__main__":
    unittest.main()
