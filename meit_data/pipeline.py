"""Run and verify the full preprocessing pipeline.

Usage:
    python -m meit_data
    python -m unittest discover -s tests -v

The output directory must not already exist. To compare another configuration,
choose a new --output and --report path. Generated arrays and observation tables
are stored in data/processed/ with their config, scaler, and provenance manifests.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import pandas as pd

from .io import assign_splits, read_events, utc_string
from .preprocess import (complete_bin_ends, event_sampling_weights,
                         event_segments, fit_scaler, make_windows, normalize_windows,
                         scaler_bin_mask, validate_config)


SPLITS = ("train", "validation", "test")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, frame):
    compression = {"method": "gzip", "mtime": 0} if str(path).endswith(".gz") else None
    frame.to_csv(path, index=False, compression=compression)


def observations_table(events):
    frames = []
    for event in events:
        frames.append(pd.DataFrame({
            "event_index": event.event_index, "event_id": event.event_id,
            "sensor_id": event.sensor_id, "source_record": event.source_record,
            "timestamp_utc": event.start + pd.to_timedelta(event.seconds, unit="s"),
            "relative_seconds": event.seconds, "depth_m": event.depth_m,
            "quality_flags": ";".join(event.flags),
            "quarantine_reasons": ";".join(event.quarantine), "split": event.split,
        }))
    return pd.concat(frames, ignore_index=True)


def sample_metadata(windows, split, events, config):
    count = len(windows["X"])
    current = windows["X"][:, -1, 0]
    future = windows["Y"][:, :, 0]
    delta = future[:, -1] - current
    phase = np.full(count, "unknown", dtype="U10")
    phase[np.isfinite(delta)] = "stable"
    phase[delta > config["evaluation"]["direction_change_threshold_m"]] = "rising"
    phase[delta < -config["evaluation"]["direction_change_threshold_m"]] = "falling"
    threshold = config["evaluation"]["threshold_depth_m"]
    fully_observed = windows["Y_mask"][:, :, 0].all(axis=1) & windows["X_observed"][:, -1, 0]
    crosses = future >= threshold
    crossing_time = np.full(count, np.nan)
    state = np.full(count, "unavailable", dtype="U20")
    state[fully_observed] = "not_reached"
    reaches = fully_observed & crosses.any(axis=1)
    state[reaches] = "reached"
    crossing_time[reaches] = crosses[reaches].argmax(axis=1) + 1
    above_now = windows["X_observed"][:, -1, 0] & (current >= threshold)
    state[above_now] = "already_above"
    crossing_time[above_now] = 0
    event_ids = windows["event_index"]
    return pd.DataFrame({
        "sample_index": np.arange(count), "split": split,
        "event_index": event_ids, "event_id": [events[i].event_id for i in event_ids],
        "sensor_id": [events[i].sensor_id for i in event_ids],
        "segment_index": windows["segment_index"],
        "input_start_bin_index": windows["input_start_bin_index"],
        "origin_time_utc": pd.to_datetime(windows["origin_time_unix_seconds"], unit="s", utc=True),
        "input_observed_bins": windows["X_observed"].sum(axis=(1, 2)),
        "target_observed_bins": windows["Y_mask"].sum(axis=(1, 2)),
        "phase_at_15_minutes": phase,
        "observed_high_depth_window": (current >= threshold) | (future >= threshold).any(axis=1),
        "threshold_evaluable_all_future_observed": fully_observed,
        "external_depth_threshold_state": state,
        "external_depth_threshold_first_bin_minutes": crossing_time,
        "sampling_weight": windows["sample_weight"],
    })


def verify_outputs(directory):
    """Independently reopen saved artifacts; reject inconsistent arrays or splits."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    manifest = json.loads((directory / "split_manifest.json").read_text())
    scaler = json.loads((directory / "scaler.json").read_text())
    bins = pd.read_csv(directory / "minute_bins.csv.gz")
    segments = pd.read_csv(directory / "segment_audit.csv").set_index("segment_index")
    samples = pd.read_csv(directory / "samples.csv.gz")
    assigned = {e["event_index"]: e for e in manifest["events"]}
    quality, window = config["quality"], config["window"]
    lookback, horizon = window["input_bins"], window["output_bins"]
    reconstructed_scaler_rows = np.zeros(len(bins), dtype=bool)
    bins_by_segment = {int(key): group for key, group in bins.groupby("segment_index", sort=False)}
    event_sets = []
    total_samples = 0
    checks = []

    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    for split in SPLITS:
        with np.load(directory / f"{split}.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
            count = len(data["X"])
            total_samples += count
            require(count > 0, f"No usable {split} samples")
            require(data["X"].shape == (count, lookback, 1), f"Bad {split} X shape")
            require(data["Y"].shape == (count, horizon, 1), f"Bad {split} Y shape")
            require(data["X"].dtype == np.float32 and data["Y"].dtype == np.float32, "Depth arrays must be float32")
            require(data["Y_mask"].dtype == bool and data["X_observed"].dtype == bool, "Masks must be boolean")
            require(data["Y_mask"].shape == data["Y"].shape, "Target mask shape mismatch")
            require(data["X_observed"].shape == data["X"].shape, "Input mask shape mismatch")
            require(data["X_age_seconds"].shape == data["X"].shape, "Observation age shape mismatch")
            require(np.isfinite(data["X"]).all(), "Nonfinite model input")
            require(np.array_equal(np.isfinite(data["Y"]), data["Y_mask"]), "Targets do not match observed mask")
            require((data["X_observed"].sum(axis=(1, 2)) >= quality["min_input_observed_bins"]).all(), "Too few input observations")
            require((data["Y_mask"].sum(axis=(1, 2)) >= quality["min_target_observed_bins"]).all(), "Too few target observations")
            require((data["X_age_seconds"] <= quality["max_last_observation_age_seconds"]).all(), "Stale observation in input")
            ids = data["event_index"]
            require(all(assigned[int(i)]["split"] == split for i in np.unique(ids)), "Event split leakage")
            event_sets.append({assigned[int(i)]["event_id"] for i in np.unique(ids)})
            origins = data["origin_time_unix_seconds"]
            require(len(np.unique(np.column_stack((ids, origins)), axis=0)) == count, "Duplicate sample origins")
            meta = samples.loc[samples.split == split]
            require(len(meta) == count, "Sample metadata count mismatch")
            require(np.array_equal(meta.event_index, ids), "Metadata event order mismatch")
            require(np.array_equal(meta.segment_index, data["segment_index"]), "Metadata segment order mismatch")
            require(np.array_equal(pd.to_datetime(meta.origin_time_utc, utc=True).astype("int64") // 10**9, origins), "Metadata origin order mismatch")
            for segment_id in np.unique(data["segment_index"]):
                selected = np.flatnonzero(data["segment_index"] == segment_id)
                segment = segments.loc[int(segment_id)]
                require((origins[selected] - lookback * 60 >= segment.first_observation_unix_seconds).all(), "Input precedes segment")
                require((origins[selected] + horizon * 60 <= segment.last_observation_unix_seconds).all(), "Target exceeds segment")
                group = bins_by_segment[int(segment_id)]
                starts = data["input_start_bin_index"][selected]
                if split == "train":
                    differences = np.zeros(len(group) + 1, dtype=np.int64)
                    np.add.at(differences, starts, 1)
                    np.add.at(differences, starts + lookback, -1)
                    covered = (differences[:-1].cumsum() > 0) & group.observed.to_numpy()
                    reconstructed_scaler_rows[group.index] = covered
                # Verify every serialized target against the original observed
                # minute table, and every input against the causal table.
                x_rows = starts[:, None] + np.arange(lookback)
                y_rows = starts[:, None] + lookback + np.arange(horizon)
                expected_x = group.causal_input_depth_m.to_numpy()[x_rows]
                expected_y = group.observed_depth_m.to_numpy()[y_rows]
                actual_x = data["X"][selected, :, 0] * scaler["std_m"] + scaler["mean_m"]
                actual_y = data["Y"][selected, :, 0] * scaler["std_m"] + scaler["mean_m"]
                require(np.allclose(actual_x, expected_x, atol=2e-7, rtol=1e-6), "Saved X differs from causal minute values")
                require(np.allclose(actual_y, expected_y, atol=2e-7, rtol=1e-6, equal_nan=True), "Saved Y differs from observed values")
            if split == "train":
                for event_id in np.unique(ids):
                    require(np.isclose(data["sample_weight"][ids == event_id].sum(), 1), "Unequal total training weight per event")
            else:
                require((data["sample_weight"] == 1).all(), "Evaluation weights must preserve original distribution")
        checks.append(f"{split}: shapes, masks, timestamps, source values and weights verified")
    require(all(not event_sets[i] & event_sets[j] for i in range(3) for j in range(i + 1, 3)), "Shared events between splits")
    require(total_samples == len(samples), "Unexpected metadata rows")
    require(np.array_equal(reconstructed_scaler_rows, bins.used_for_scaler.to_numpy()), "Scaler fit membership differs from unique training inputs")
    fitted = bins.observed_depth_m.to_numpy()[reconstructed_scaler_rows]
    require(len(fitted) == scaler["unique_observed_bin_count"], "Scaler observation count mismatch")
    require(np.isclose(fitted.mean(), scaler["mean_m"], rtol=0, atol=1e-12), "Scaler mean mismatch")
    require(np.isclose(fitted.std(), scaler["std_m"], rtol=0, atol=1e-12), "Scaler standard deviation mismatch")
    for previous, following in zip(SPLITS[:-1], SPLITS[1:]):
        left = [pd.Timestamp(e["effective_end_utc"]) for e in manifest["events"] if e["split"] == previous]
        right = [pd.Timestamp(e["start_utc"]) for e in manifest["events"] if e["split"] == following]
        require(max(left) < min(right), "Overlapping split date ranges")
    checks.extend(["unique observed training bins independently reconciled with scaler",
                   "all split events disjoint and chronological", "all saved samples reconciled with minute tables"])
    data_manifest = directory / "data_manifest.json"
    if data_manifest.exists():
        artifacts = json.loads(data_manifest.read_text())["artifacts"]
        for name, expected in artifacts.items():
            require(sha256(directory / name) == expected["sha256"], f"Artifact hash mismatch: {name}")
        checks.append("all recorded artifact SHA-256 hashes verified")
    return {"passed": True, "checks": checks}


def run_pipeline(source, config_path, output, report):
    source, config_path, output, report = map(Path, (source, config_path, output, report))
    if output.exists() or report.exists():
        raise FileExistsError("Choose new --output and --report paths; existing results are not overwritten")
    if report.resolve().is_relative_to(output.resolve()):
        raise ValueError("--report must be outside --output")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    source_hash = sha256(source)
    events, audit, raw_info = read_events(source, config)
    manifest = assign_splits(events, config)
    split_by_record = {e.source_record: e for e in events}
    audit["split"] = [split_by_record[r].split if r in split_by_record else "excluded" for r in audit.source_record]
    audit["split_exclusion_reason"] = [split_by_record[r].split_reason if r in split_by_record else "invalid_or_duplicate_record" for r in audit.source_record]
    chunks = {split: [] for split in SPLITS}
    stats = {split: Counter() for split in SPLITS}
    minute_frames, segment_records, train_values = [], [], []
    segment_index = 0
    for event in events:
        if event.split == "excluded":
            continue
        counters = stats[event.split]
        counters["assigned_events"] += 1
        full_bins = len(complete_bin_ends(event.seconds[0], event.seconds[-1]))
        counters["candidate_windows_before_gap_splitting"] += max(0, (full_bins - 45) // config["window"]["stride_bins"] + 1)
        event_window_count = 0
        point_offset = 0
        for segment_number, point_count, bins in event_segments(event, config):
            windows, segment_stats = make_windows(bins, config)
            counters.update(segment_stats)
            counters["segments"] += 1
            counters["segments_too_short_for_45_bins"] += int(len(bins) < 45)
            count = len(windows["X"])
            event_window_count += count
            bins["event_index"], bins["segment_index"], bins["split"] = event.event_index, segment_index, event.split
            bins["used_for_scaler"] = False
            if event.split == "train" and count:
                scaler_mask = scaler_bin_mask(bins, windows, config["window"]["input_bins"])
                bins["used_for_scaler"] = scaler_mask
                train_values.append(bins.observed_depth_m.to_numpy()[scaler_mask])
            if count:
                windows["event_index"] = np.full(count, event.event_index, dtype=np.int32)
                windows["segment_index"] = np.full(count, segment_index, dtype=np.int32)
                chunks[event.split].append(windows)
            # Segment raw span is recovered from its original point counts.
            first_second = event.seconds[point_offset]
            last_second = event.seconds[point_offset + point_count - 1]
            point_offset += point_count
            segment_records.append({
                "segment_index": segment_index, "event_index": event.event_index,
                "event_id": event.event_id, "segment_number_in_event": segment_number,
                "split": event.split, "raw_point_count": int(point_count),
                "first_observation_unix_seconds": int(event.start.timestamp()) + float(first_second),
                "last_observation_unix_seconds": int(event.start.timestamp()) + float(last_second),
                **segment_stats,
            })
            if len(bins):
                minute_frames.append(bins)
            segment_index += 1
        counters["events_with_samples"] += int(event_window_count > 0)
        counters["events_without_samples"] += int(event_window_count == 0)
    if not train_values:
        raise ValueError("No usable training input observations after quality filtering")
    scaler = fit_scaler(np.concatenate(train_values))
    minute_table = pd.concat(minute_frames, ignore_index=True)
    segments_table = pd.DataFrame(segment_records)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".preprocess-", dir=output.parent))
    try:
        write_json(stage / "config.json", config)
        write_json(stage / "scaler.json", scaler)
        write_json(stage / "split_manifest.json", manifest)
        write_csv(stage / "event_audit.csv", audit)
        write_csv(stage / "segment_audit.csv", segments_table)
        write_csv(stage / "observations.csv.gz", observations_table(events))
        write_csv(stage / "minute_bins.csv.gz", minute_table)
        all_metadata, split_summary = [], {}
        for split in SPLITS:
            if not chunks[split]:
                raise ValueError(f"No usable {split} samples; inspect date boundaries and quality limits")
            windows = {key: np.concatenate([chunk[key] for chunk in chunks[split]]) for key in chunks[split][0]}
            count = len(windows["X"])
            windows["sample_weight"] = event_sampling_weights(windows["event_index"]) if split == "train" else np.ones(count)
            metadata = sample_metadata(windows, split, events, config)
            all_metadata.append(metadata)
            np.savez_compressed(stage / f"{split}.npz", **normalize_windows(windows, scaler))
            assigned_events = [e for e in events if e.split == split]
            represented = {int(i) for i in np.unique(windows["event_index"])}
            split_summary[split] = {
                **dict(stats[split]), "X_shape": [count, 30, 1], "Y_shape": [count, 15, 1],
                "source_start_utc": utc_string(min(e.start for e in assigned_events)),
                "source_end_utc": utc_string(max(e.effective_end for e in assigned_events)),
                "sensors_with_samples": len({events[i].sensor_id for i in represented}),
                "high_depth_events_with_samples": sum(events[i].depth_m.max() >= config["evaluation"]["threshold_depth_m"] for i in represented),
                "observed_high_depth_windows": int(metadata.observed_high_depth_window.sum()),
                "phase_counts": {str(k): int(v) for k, v in metadata.phase_at_15_minutes.value_counts().items()},
                "observed_targets_by_horizon": windows["Y_mask"].sum(axis=(0, 2)).tolist(),
                "threshold_state_counts": {str(k): int(v) for k, v in metadata.external_depth_threshold_state.value_counts().items()},
                "unique_events_reaching_threshold_from_below": int(metadata.loc[metadata.external_depth_threshold_state == "reached", "event_index"].nunique()),
            }
        write_csv(stage / "samples.csv.gz", pd.concat(all_metadata, ignore_index=True))
        verification = verify_outputs(stage)
        if sha256(source) != source_hash:
            raise RuntimeError("Source CSV changed during processing")
        implementation = hashlib.sha256()
        for path in sorted(Path(__file__).parent.glob("*.py")):
            implementation.update(path.name.encode())
            implementation.update(path.read_bytes())
        summary = {
            "source": source.name, "source_sha256": source_hash,
            "config_sha256": sha256(config_path), "implementation_sha256": implementation.hexdigest(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "contract": {"input_minutes": 30, "forecast_minutes": list(range(1, 16)),
                         "bin_seconds": 60, "depth_unit_before_scaling": "m", "saved_depth_arrays": "standardized_float32"},
            "raw": {**raw_info, "point_count_before_dedup": int(audit.raw_point_count.sum()),
                    "point_count_after_dedup": int(audit.retained_point_count.sum()),
                    "exact_duplicate_points_removed": int(audit.exact_duplicate_points_removed.sum()),
                    "event_status_counts": {str(k): int(v) for k, v in audit.status.value_counts().items()},
                    "summary_peak_mismatch_events": int(audit["flags"].str.contains("summary_peak_mismatch").sum()),
                    "high_depth_events_before_quarantine": sum(e.depth_m.max() >= config["evaluation"]["threshold_depth_m"] for e in events)},
            "exclusion_counts": {str(k): int(v) for k, v in audit.loc[audit.split == "excluded", "split_exclusion_reason"].value_counts().items()},
            "cut_dates_utc": manifest["cut_dates_utc"], "scaler": scaler,
            "splits": split_summary, "verification": verification,
            "limits": [
                "Event exports omit continuous dry periods and pre-event history; no zero-padding was added.",
                f"Summary peak differences above {config['quality']['summary_peak_quarantine_difference_m']} m are conservatively quarantined pending review.",
                f"Same-date grouping and {config['split']['embargo_after_cut_hours']}-hour embargo do not establish storm independence.",
                "Targets are the last observed depth in each future minute bin; missing labels remain NaN.",
                f"External depth threshold {config['evaluation']['threshold_depth_m']:.2f} m is a project screening assumption, not a label for door opening or evacuation.",
                "Observation timestamps are assumed available without network latency; received timestamps are absent.",
                "Source event selection and retrospective quality filtering can bias continuous-stream performance.",
            ],
        }
        if split_summary["test"]["unique_events_reaching_threshold_from_below"] == 0:
            summary["limits"].append(
                "Test split has zero evaluable from-below threshold crossings with fully observed future labels; "
                "threshold-onset recall and arrival-time error cannot be estimated on this split."
            )
        # All counters must be built-in scalars for strict JSON serialization.
        summary = json.loads(json.dumps(summary, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=False))
        artifacts = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                     for path in sorted(stage.iterdir()) if path.is_file()}
        write_json(stage / "data_manifest.json", {
            **summary, "artifacts": artifacts,
            "array_schema": {
                "X": "float32 (N,30,1), standardized causal depth",
                "Y": "float32 (N,15,1), standardized observed depth; NaN where Y_mask is false",
                "Y_mask": "bool (N,15,1); select valid positions BEFORE loss/metrics, not NaN multiplied by zero",
                "X_observed": "bool (N,30,1); imputed input bins remain false",
                "X_age_seconds": "float32 (N,30,1), age of last real observation",
                "origin_time_unix_seconds": "int64 (N,), UTC end of the last input bin",
                "event_index": "int32 (N,), maps to split_manifest.json",
                "segment_index": "int32 (N,), maps to segment_audit.csv",
                "input_start_bin_index": "int64 (N,), zero-based within segment minute table",
                "sample_weight": "float64 (N,), train=1/event sample count; validation/test=1",
            },
            "sampling_weight_usage": "Use training weights as sampling probabilities OR loss weights, not both. Evaluation remains unweighted.",
            "time_semantics": "Bin (T-60s,T] includes right endpoint only. Keep event-start grid across gaps. Omit partial bins outside segment spans.",
        })
        report.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(output)
        write_json(report, summary)
        return summary
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description="Preprocess FloodNet: 30-minute inputs, all 15 future minute targets")
    parser.add_argument("--source", type=Path, default=Path("export.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline_config.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--report", type=Path, default=Path("analysis/preprocessing_summary.json"))
    parser.add_argument("--verify-only", action="store_true", help="Reopen and validate existing --output artifacts")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_outputs(args.output), ensure_ascii=False, indent=2))
        return
    summary = run_pipeline(args.source, args.config, args.output, args.report)
    print(json.dumps({"output": str(args.output), "report": str(args.report),
                      "cut_dates_utc": summary["cut_dates_utc"],
                      "samples": {split: summary["splits"][split]["accepted_windows"] for split in SPLITS},
                      "exclusions": summary["exclusion_counts"],
                      "verified": summary["verification"]["passed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
