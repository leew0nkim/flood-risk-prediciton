import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from meit_model.final_evaluation_gate import validate_fresh_holdout, verify_release


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FinalEvaluationGateTests(unittest.TestCase):
    def test_release_hash_and_fresh_event_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for split, event in zip(
                    ("train", "validation", "test_risk", "test_temporal"),
                    (1, 2, 3, 4)):
                np.savez(data / f"{split}.npz", event_index=np.array([event]),
                         origin_time_unix_seconds=np.array([100 + event]))
            marker = data / "scaler.json"
            marker.write_text("{}")
            contract = {
                "schema_version": 1,
                "release_id": "test",
                "artifact_hashes": {"scaler.json": {"sha256": digest(marker)}},
                "development_fingerprint": {},
                "final_holdout": {"status": "REQUIRED_NEW_UNOPENED_DATA"},
            }
            self.assertEqual(verify_release(contract, data)["release_id"], "test")
            marker.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_release(contract, data)

            holdout = root / "fresh_final_holdout.npz"
            arrays = dict(
                X=np.zeros((1, 30, 1), np.float32),
                Y=np.zeros((1, 15, 1), np.float32),
                X_observed=np.ones((1, 30, 1), bool),
                Y_mask=np.ones((1, 15, 1), bool),
                event_index=np.array([99]),
                origin_time_unix_seconds=np.array([999]),
            )
            np.savez(holdout, **arrays)
            contract["final_holdout"] = {
                "status": "SEALED_UNOPENED", "registered_path": holdout.name,
                "sha256": digest(holdout),
            }
            self.assertEqual(
                validate_fresh_holdout(contract, data, holdout)["event_count"], 1)
            arrays["event_index"] = np.array([1])
            np.savez(holdout, **arrays)
            contract["final_holdout"]["sha256"] = digest(holdout)
            with self.assertRaisesRegex(ValueError, "reuses 1 existing event"):
                validate_fresh_holdout(contract, data, holdout)

    def test_unregistered_holdout_is_blocked(self):
        contract = {"final_holdout": {"status": "REQUIRED_NEW_UNOPENED_DATA"}}
        with self.assertRaisesRegex(ValueError, "fresh holdout"):
            validate_fresh_holdout(contract, Path("unused"), Path("unused.npz"))


if __name__ == "__main__":
    unittest.main()
