"""Verify the frozen data release and admit only a registered fresh holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from meit_data.pipeline import sha256


EXISTING_SPLITS = ("train", "validation", "test_risk", "test_temporal")
REQUIRED_ARRAYS = {
    "X", "Y", "X_observed", "Y_mask", "origin_time_unix_seconds", "event_index"
}


def read_contract(path: Path) -> dict:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported data-release contract schema")
    return contract


def verify_release(contract: dict, data_dir: Path, source: Path | None = None) -> dict:
    data_dir = Path(data_dir)
    for name, record in contract.get("artifact_hashes", {}).items():
        path = data_dir / name
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise ValueError(f"Canonical data-release hash mismatch: {name}")
    if source is not None:
        expected = contract.get("source_csv", {}).get("sha256")
        if not Path(source).is_file() or sha256(source) != expected:
            raise ValueError("Canonical source CSV hash mismatch")
    return {
        "release_id": contract["release_id"],
        "artifact_count": len(contract["artifact_hashes"]),
        "development_fingerprint": contract["development_fingerprint"],
    }


def validate_fresh_holdout(contract: dict, data_dir: Path, holdout: Path) -> dict:
    registration = contract.get("final_holdout", {})
    if registration.get("status") != "SEALED_UNOPENED":
        raise ValueError("Final evaluation is blocked: a fresh holdout is not registered as SEALED_UNOPENED")
    holdout = Path(holdout)
    registered = registration.get("registered_path")
    if not registered or Path(registered).name != holdout.name:
        raise ValueError("Holdout path does not match the frozen registration")
    if not holdout.is_file() or sha256(holdout) != registration.get("sha256"):
        raise ValueError("Registered holdout SHA-256 mismatch")
    if holdout.name in {f"{split}.npz" for split in EXISTING_SPLITS}:
        raise ValueError("Existing exploratory splits cannot be reused as the fresh final holdout")

    with np.load(holdout, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS - set(archive.files)
        if missing:
            raise ValueError(f"Fresh holdout lacks required arrays: {sorted(missing)}")
        event_ids = np.asarray(archive["event_index"])
        origins = np.asarray(archive["origin_time_unix_seconds"])
        if event_ids.ndim != 1 or origins.shape != event_ids.shape or not len(event_ids):
            raise ValueError("Fresh holdout has invalid event/origin arrays")
        if len(np.unique(np.column_stack((event_ids, origins)), axis=0)) != len(event_ids):
            raise ValueError("Fresh holdout contains duplicate event/origin samples")

    prior_events = set()
    for split in EXISTING_SPLITS:
        with np.load(Path(data_dir) / f"{split}.npz", allow_pickle=False) as archive:
            prior_events.update(map(int, np.unique(archive["event_index"])))
    overlap = prior_events & set(map(int, np.unique(event_ids)))
    if overlap:
        raise ValueError(f"Fresh holdout reuses {len(overlap)} existing event IDs")
    return {
        "status": "READY_FOR_ONE_FINAL_EVALUATION",
        "holdout": str(holdout),
        "sha256": sha256(holdout),
        "sample_count": int(len(event_ids)),
        "event_count": int(len(np.unique(event_ids))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path,
                        default=Path("configs/canonical_data_release.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--source", type=Path, default=Path("export.csv"))
    parser.add_argument("--holdout", type=Path)
    args = parser.parse_args(argv)
    contract = read_contract(args.release)
    result = {"release": verify_release(contract, args.data_dir, args.source)}
    if args.holdout is None:
        result["final_holdout"] = contract["final_holdout"]
    else:
        result["final_holdout"] = validate_fresh_holdout(
            contract, args.data_dir, args.holdout)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
