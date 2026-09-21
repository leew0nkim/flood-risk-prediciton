import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from meit_data.io import DEPTH, DRAIN_TIME, DURATION, END, PEAK, PEAK_TIME, SECONDS, SENSOR, START
from meit_data.pipeline import SPLITS, run_pipeline, sha256, verify_outputs
from test_io import config


class PipelineIntegrationTests(unittest.TestCase):
    def fixture(self, root):
        rows = []
        for day in range(1, 31):
            start = pd.Timestamp(f"2024-01-{day:02}T00:00:00Z")
            seconds = np.delete(np.arange(61) * 60, [4, 35])
            if day in (2, 5, 8, 11, 14, 17, 20):
                depths = np.round(np.linspace(0, 20, len(seconds)), 4)
            else:
                depths = np.round(np.arange(len(seconds)) * .1 + day * .01, 4)
            row = {SENSOR: f"sensor_{day % 3}", START: start.strftime("%m/%d/%Y %I:%M:%S %p"),
                   END: (start + pd.Timedelta(hours=1)).strftime("%m/%d/%Y %I:%M:%S %p"),
                   DEPTH: json.dumps(depths.tolist()), SECONDS: json.dumps(seconds.tolist()),
                   PEAK: str(depths.max()), PEAK_TIME: "60", DRAIN_TIME: "0", DURATION: "60"}
            rows.append(row)
        source = root / "source.csv"
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        settings = root / "config.json"
        settings_config = config()
        settings_config["split"]["risk_test"]["selected_dates_utc"] = [
            f"2024-01-{day:02}T00:00:00Z" for day in (5, 8, 17, 20)
        ]
        settings.write_text(json.dumps(settings_config))
        return source, settings

    def test_full_pipeline_reopens_and_reproduces_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, settings = self.fixture(root)
            original_hash = sha256(source)
            first = run_pipeline(source, settings, root / "first", root / "first.json")
            second = run_pipeline(source, settings, root / "second", root / "second.json")
            self.assertTrue(first["verification"]["passed"])
            self.assertEqual(first, second)
            self.assertEqual(original_hash, sha256(source))
            self.assertTrue(verify_outputs(root / "first")["passed"])
            self.assertGreaterEqual(first["splits"]["test_risk"]["unique_events_reaching_threshold_from_below"], 4)
            warning_states = first["splits"]["test_risk"]["warning_threshold_state_counts"]
            self.assertGreater(warning_states.get("reached", 0) + warning_states.get("already_above", 0), 0)
            self.assertEqual(len(first["risk_test"]["selected_dates_utc"]), 4)
            self.assertEqual(first["risk_test"]["critical_threshold_m"], .3)
            self.assertEqual(first["risk_test"]["selection_procedure"]["chronological_strata"], 4)
            for split in SPLITS:
                with np.load(root / "first" / f"{split}.npz") as a, np.load(root / "second" / f"{split}.npz") as b:
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
                    self.assertGreater(len(a["X"]), 0)
                    self.assertEqual(a["Y"].shape[1:], (15, 1))
            self.assertTrue((root / "first" / "data_manifest.json").is_file())
            data_manifest = json.loads((root / "first" / "data_manifest.json").read_text())
            self.assertEqual(data_manifest["evaluation_protocol"]["warning_threshold_m"], .25)
            self.assertEqual(data_manifest["evaluation_protocol"]["critical_threshold_m"], .3)
            self.assertEqual(data_manifest["risk_metadata_schema"]["warning"]["threshold_m"], .25)
            self.assertEqual(data_manifest["risk_metadata_schema"]["critical"]["threshold_m"], .3)
            self.assertEqual(
                data_manifest["risk_metadata_schema"]["warning"]["observed_high_depth_column"],
                "warning_observed_high_depth_window",
            )
            self.assertEqual(
                data_manifest["risk_metadata_schema"]["critical"]["observed_high_depth_column"],
                "observed_high_depth_window",
            )
            with self.assertRaises(FileExistsError):
                run_pipeline(source, settings, root / "first", root / "other.json")

    def test_saved_target_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, settings = self.fixture(root)
            run_pipeline(source, settings, root / "output", root / "report.json")
            path = root / "output" / "test_temporal.npz"
            with np.load(path) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["Y"][0, 0, 0] += 100
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(AssertionError, "Saved Y differs"):
                verify_outputs(root / "output")

    def test_saved_risk_metadata_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, settings = self.fixture(root)
            run_pipeline(source, settings, root / "output", root / "report.json")
            path = root / "output" / "samples.csv.gz"
            samples = pd.read_csv(path)
            samples.loc[0, "external_depth_threshold_state"] = "reached"
            samples.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            with self.assertRaisesRegex(AssertionError, "threshold state metadata mismatch"):
                verify_outputs(root / "output")

    def test_heldout_depth_changes_do_not_change_training_or_scaler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, settings = self.fixture(root)
            first = run_pipeline(source, settings, root / "first", root / "first.json")
            cutoff = pd.Timestamp(first["cut_dates_utc"][0])
            rows = pd.read_csv(source, dtype=str, keep_default_na=False)
            for index, row in rows.iterrows():
                if pd.to_datetime(row[START], format="%m/%d/%Y %I:%M:%S %p", utc=True) >= cutoff:
                    depths = np.asarray(json.loads(row[DEPTH])) * 10
                    rows.at[index, DEPTH] = json.dumps(depths.tolist())
                    rows.at[index, PEAK] = str(depths.max())
            changed_source = root / "changed.csv"
            rows.to_csv(changed_source, index=False)
            second = run_pipeline(changed_source, settings, root / "second", root / "second.json")
            self.assertEqual(first["scaler"], second["scaler"])
            self.assertEqual(first["cut_dates_utc"], second["cut_dates_utc"])
            with np.load(root / "first" / "train.npz") as a, np.load(root / "second" / "train.npz") as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
