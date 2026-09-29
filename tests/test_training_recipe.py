import json
from pathlib import Path
import tempfile
import unittest

from meit_model.risk_policy import (
    EXTERNAL_EVALUATION_SCOPE,
    REPORT_SCHEMA_VERSION,
    create_policy_decision,
    load_policy_decision,
    threshold_contract,
)
from meit_model.cv import CV_SCHEMA_VERSION
from meit_model.train_final_risk_lstm import derive_training_recipe


class TrainingRecipeTests(unittest.TestCase):
    def write_json(self, path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def evidence(self, root, fingerprint, threshold=.29):
        window = root / "window.json"
        event = root / "event.json"
        model_evidence = [{"fold": fold, "checkpoint": f"fold_{fold}/best_model.pt",
                           "sha256": str(fold) * 64} for fold in range(1, 5)]
        self.write_json(window, {
            "schema_version": REPORT_SCHEMA_VERSION,
            "report_type": "window_calibration",
            "evaluation_scope": EXTERNAL_EVALUATION_SCOPE,
            "sealed_test_used": False,
            "threshold_contract": threshold_contract(),
            "data_fingerprint": fingerprint,
            "actual_critical_depth_m": .30,
            "model_evidence": model_evidence,
            "lstm_threshold_sweep": [{
                "alarm_threshold_m": threshold,
                "pooled": {"timely_recall": .8, "recall": .9,
                           "false_positive_rate": .1, "precision": .7,
                           "alarm_rate": .2},
            }],
        })
        self.write_json(event, {
            "schema_version": REPORT_SCHEMA_VERSION,
            "report_type": "event_evaluation",
            "evaluation_scope": EXTERNAL_EVALUATION_SCOPE,
            "sealed_test_used": False,
            "threshold_contract": threshold_contract(),
            "data_fingerprint": fingerprint,
            "actual_critical_depth_m": .30,
            "required_lead_time_min": 5,
            "model_evidence": model_evidence,
            "pooled": [{
                "warning_threshold_m": threshold,
                "summary": {"operational_event_recall": .75,
                            "required_lead_time_min": 5},
            }],
        })
        return window, event

    def cv_summaries(self, root, fingerprint):
        epoch_groups = ([39, 15, 17, 21], [28, 15, 14, 27], [33, 11, 17, 20])
        summaries = []
        for seed, epochs in zip((42, 43, 44), epoch_groups):
            path = root / f"seed_{seed}.json"
            self.write_json(path, {
                "schema_version": CV_SCHEMA_VERSION,
                "data_fingerprint": fingerprint,
                "evaluation_scope": "development_cv_validation",
                "sealed_test_used": False,
                "critical_threshold_role": "external_depth_crossing",
                "cv_split_summary_sha256": "a" * 64,
                "seed": seed,
                "config": {"critical_depth_m": .30},
                "folds": [{"best_epoch": epoch} for epoch in epochs],
            })
            summaries.append(path)
        return summaries

    def test_recipe_requires_reviewed_window_and_event_evidence(self):
        fingerprint = {"train.npz": "current"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summaries = self.cv_summaries(root, fingerprint)
            window, event = self.evidence(root, fingerprint)
            decision = root / "policy.json"
            self.write_json(decision, create_policy_decision(
                window, event, fingerprint, .29,
                "Selected after comparing 5-minute event recall, FPR, precision and alarm rate."))

            recipe = derive_training_recipe(summaries, decision, window, event, fingerprint)
            self.assertEqual(recipe["epochs"], 19)
            self.assertEqual(recipe["forecast_warning_depth_m"], .29)
            self.assertEqual(recipe["cv_seeds"], [42, 43, 44])
            self.assertEqual(recipe["policy_decision"]["selection_basis"]["primary_metric"],
                             "operational_event_recall_at_5_min")
            self.assertTrue(recipe["policy_decision"]["threshold_contract"]
                            ["same_numeric_value_but_distinct_metrics"])

    def test_policy_rejects_stale_data_and_tampered_evidence(self):
        fingerprint = {"train.npz": "current"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window, event = self.evidence(root, fingerprint)
            decision = root / "policy.json"
            self.write_json(decision, create_policy_decision(
                window, event, fingerprint, .29, "Reviewed all required metrics."))
            with self.assertRaisesRegex(ValueError, "different data"):
                load_policy_decision(decision, window, event, {"train.npz": "newer"})
            payload = json.loads(window.read_text())
            payload["lstm_threshold_sweep"][0]["pooled"]["alarm_rate"] = .9
            self.write_json(window, payload)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_policy_decision(decision, window, event, fingerprint)

    def test_five_minute_event_report_is_mandatory(self):
        fingerprint = {"train.npz": "current"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window, event = self.evidence(root, fingerprint)
            payload = json.loads(event.read_text())
            payload["required_lead_time_min"] = 4
            self.write_json(event, payload)
            with self.assertRaisesRegex(ValueError, "5-minute"):
                create_policy_decision(window, event, fingerprint, .29, "Reviewed metrics.")

    def test_recipe_rejects_noncanonical_critical_depth(self):
        fingerprint = {"train.npz": "current"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summaries = self.cv_summaries(root, fingerprint)
            payload = json.loads(summaries[0].read_text())
            payload["config"]["critical_depth_m"] = .40
            self.write_json(summaries[0], payload)
            window, event = self.evidence(root, fingerprint)
            decision = root / "policy.json"
            self.write_json(decision, create_policy_decision(
                window, event, fingerprint, .29, "Reviewed metrics."))
            with self.assertRaisesRegex(ValueError, "must be 0.30 m"):
                derive_training_recipe(summaries, decision, window, event, fingerprint)

    def test_policy_rejects_report_that_does_not_declare_sealed_test_status(self):
        fingerprint = {"train.npz": "current"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window, event = self.evidence(root, fingerprint)
            payload = json.loads(event.read_text())
            payload.pop("sealed_test_used")
            self.write_json(event, payload)
            with self.assertRaisesRegex(ValueError, "development CV validation only"):
                create_policy_decision(window, event, fingerprint, .29, "Reviewed metrics.")


if __name__ == "__main__":
    unittest.main()
