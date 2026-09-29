import json
from pathlib import Path
import tempfile
import unittest

from meit_model.calibrate_risk_warning import select_warning_threshold
from meit_model.train_final_risk_lstm import derive_training_recipe


class TrainingRecipeTests(unittest.TestCase):
    def write_json(self, path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def test_recipe_is_derived_from_current_cv_and_calibration(self):
        fingerprint = {"train.npz": "current"}
        epoch_groups = ([39, 15, 17, 21], [28, 15, 14, 27], [33, 11, 17, 20])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summaries = []
            for seed, epochs in zip((42, 43, 44), epoch_groups):
                path = root / f"seed_{seed}.json"
                self.write_json(path, {
                    "schema_version": 2,
                    "data_fingerprint": fingerprint,
                    "seed": seed,
                    "folds": [{"best_epoch": epoch} for epoch in epochs],
                })
                summaries.append(path)
            calibration = root / "calibration.json"
            self.write_json(calibration, {
                "schema_version": 2,
                "data_fingerprint": fingerprint,
                "actual_critical_depth_m": .30,
                "selection": {"selected_threshold_m": .29, "rule": "validation-only"},
            })

            recipe = derive_training_recipe(summaries, calibration, fingerprint)
            self.assertEqual(recipe["epochs"], 19)
            self.assertEqual(recipe["forecast_warning_depth_m"], .29)
            self.assertEqual(recipe["cv_seeds"], [42, 43, 44])

    def test_recipe_rejects_stale_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summaries = []
            for seed in (42, 43, 44):
                path = root / f"seed_{seed}.json"
                self.write_json(path, {
                    "schema_version": 2,
                    "data_fingerprint": {"train.npz": "old"},
                    "seed": seed,
                    "folds": [{"best_epoch": 1}] * 4,
                })
                summaries.append(path)
            calibration = root / "calibration.json"
            self.write_json(calibration, {
                "schema_version": 2,
                "data_fingerprint": {"train.npz": "current"},
                "actual_critical_depth_m": .30,
                "selection": {"selected_threshold_m": .29, "rule": "validation-only"},
            })
            with self.assertRaisesRegex(ValueError, "different data"):
                derive_training_recipe(summaries, calibration, {"train.npz": "current"})

    def test_threshold_selection_is_safety_first_and_deterministic(self):
        results = [
            {"alarm_threshold_m": .30, "pooled": {
                "timely_recall": .7, "recall": .8,
                "false_positive_rate": .01, "precision": .9,
            }},
            {"alarm_threshold_m": .29, "pooled": {
                "timely_recall": .8, "recall": .85,
                "false_positive_rate": .10, "precision": .7,
            }},
        ]
        self.assertEqual(select_warning_threshold(results)["alarm_threshold_m"], .29)


if __name__ == "__main__":
    unittest.main()
