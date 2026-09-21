"""Revalidate saved data against raw CSV, streaming inputs and the planned LSTM.

Run with a Python environment containing the optional torch dependency:
    python -m meit_data.lstm_check

The optimizer runs eight training-only batches to test connectivity and finite
gradients. No fitted model is saved and no evaluation data updates weights.
The report is an integration audit, not a forecast-performance claim.
"""

import argparse
import json
from pathlib import Path
import platform
import unicodedata

import numpy as np
import pandas as pd
import torch

from .inference_input import prepare_input
from .io import DEPTH, SECONDS, assign_splits, read_events
from .pipeline import SPLITS, assign_risk_holdout, sha256, verify_outputs, write_json
from .preprocess import complete_bin_ends, event_segments, make_windows
from .torch_data import ForecastDataset, depth_to_m, horizon_errors, make_loader, masked_mae


class ReferenceLSTM(torch.nn.Module):
    """Exact architecture in docs/forecast_pipeline.md, for compatibility checks."""

    def __init__(self, config):
        super().__init__()
        settings = config["model"]
        self.lstm = torch.nn.LSTM(input_size=settings["input_size"], hidden_size=settings["hidden_size"],
                                  num_layers=settings["num_layers"], batch_first=settings["batch_first"])
        self.head = torch.nn.Linear(settings["hidden_size"], settings["head_output_size"])

    def forward(self, inputs):
        hidden, _ = self.lstm(inputs)
        return self.head(hidden[:, -1]).unsqueeze(-1)


def raw_and_streaming_checks(source, directory, datasets):
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    manifest = json.loads((directory / "split_manifest.json").read_text())
    assignments = {item["event_index"]: item for item in manifest["events"]}
    cache, results = {}, {}
    for split, dataset in datasets.items():
        arrays = {key: value.numpy() for key, value in dataset.arrays.items()}
        segment_ids, first_indices, counts = np.unique(arrays["segment_index"], return_index=True, return_counts=True)
        chosen = np.unique(np.r_[first_indices, first_indices + counts - 1])
        checked = 0
        for index in chosen:
            event_index = int(arrays["event_index"][index])
            if event_index not in cache:
                assignment = assignments[event_index]
                row = raw.iloc[assignment["source_record"] - 2]
                seconds = np.asarray(json.loads(row[SECONDS]), dtype=np.float64)
                depths = np.asarray(json.loads(row[DEPTH]), dtype=np.float64) * .0254
                start = int(pd.Timestamp(assignment["start_utc"]).timestamp())
                cache[event_index] = (seconds, depths, start)
            seconds, depths, start = cache[event_index]
            issued_at = int(arrays["origin_time_unix_seconds"][index])
            relative_issue = issued_at - start
            truth = []
            # Independent right-closed interval selection, not the production
            # searchsorted/minute-table implementation.
            for minute in range(1, 16):
                end = relative_issue + minute * 60
                inside = (seconds > end - 60) & (seconds <= end)
                truth.append(float(depths[inside][-1]) if inside.any() else np.nan)
            truth = np.asarray(truth)
            actual = arrays["Y"][index, :, 0].astype(np.float64) * dataset.scaler["std_m"] + dataset.scaler["mean_m"]
            np.testing.assert_allclose(actual, truth, atol=2e-7, rtol=1e-6, equal_nan=True)
            np.testing.assert_array_equal(arrays["Y_mask"][index, :, 0], np.isfinite(truth))
            past = seconds + start <= issued_at
            live = prepare_input(seconds[past] + start, depths[past], issued_at, start, dataset.config, dataset.scaler)
            if live["status"] != "READY":
                raise AssertionError(f"Stored sample is not READY from its historical prefix: {event_index}, {index}")
            np.testing.assert_array_equal(live["X"][0], arrays["X"][index])
            np.testing.assert_array_equal(live["X_observed"][0], arrays["X_observed"][index])
            np.testing.assert_array_equal(live["X_age_seconds"][0], arrays["X_age_seconds"][index])
            checked += 1
        results[split] = {"segments_checked": len(segment_ids), "samples_checked": checked,
                          "target_positions_checked_directly_against_csv": checked * 15,
                          "raw_target_and_mask_match": True, "past_only_streaming_input_exact_match": True}
    return results


def optimizer_check(datasets):
    config = datasets["train"].config
    torch.manual_seed(config["training"]["seed"])
    model = ReferenceLSTM(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    before = torch.cat([parameter.detach().flatten().clone() for parameter in model.parameters()])
    losses, valid_labels = [], []
    model.train()
    for step, batch in enumerate(make_loader(datasets["train"])):
        if step == 8:
            break
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["X"])
        if prediction.shape != batch["Y"].shape:
            raise AssertionError("Reference LSTM output and saved labels disagree")
        loss = masked_mae(prediction, batch["Y"], batch["Y_mask"])
        loss.backward()
        if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise AssertionError("Missing or nonfinite parameter gradients")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
        if not torch.isfinite(norm):
            raise AssertionError("Nonfinite gradient norm")
        optimizer.step()
        losses.append(float(loss.detach()))
        valid_labels.append(int(batch["Y_mask"].sum()))
    after = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    if torch.equal(before, after) or not torch.isfinite(after).all():
        raise AssertionError("Optimizer did not produce finite parameter updates")
    frozen_parameters = after.clone()
    evaluation = {}
    model.eval()
    with torch.no_grad():
        for split in ("validation", "test_temporal", "test_risk"):
            batch = next(iter(make_loader(datasets[split])))
            output = model(batch["X"])
            predicted_m = depth_to_m(output, datasets[split].scaler, clip_negative=True)
            observed_m = depth_to_m(batch["Y"], datasets[split].scaler)
            metrics = horizon_errors(predicted_m, observed_m, batch["Y_mask"])
            if not torch.isfinite(predicted_m).all() or not torch.isfinite(metrics["absolute_error_sum_m"]).all():
                raise AssertionError("Invalid inverse scaling or masked evaluation")
            evaluation[split] = {"batch_shape": list(batch["X"].shape), "output_shape": list(output.shape),
                                 "finite_predictions": True, "observed_targets_by_horizon": metrics["observed_count"].tolist()}
    if not torch.equal(frozen_parameters, torch.cat([parameter.detach().flatten() for parameter in model.parameters()])):
        raise AssertionError("Evaluation changed model weights")
    return {"purpose": "integration_check_only_not_performance_estimation", "model": config["model"],
            "optimizer": "Adam", "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"], "training_steps": len(losses),
            "training_samples_drawn_with_replacement": len(losses) * config["training"]["batch_size"],
            "standardized_masked_training_losses": losses, "observed_labels_per_step": valid_labels,
            "finite_gradients_and_parameter_updates": True, "evaluation_splits_do_not_update_weights": True,
            "model_weights_saved": False, "evaluation_shape_checks": evaluation}


def coverage_and_quarantine(source, directory, datasets):
    config = datasets["train"].config
    events, _, _ = read_events(source, config)
    generated_manifest = assign_risk_holdout(events, config, assign_splits(events, config))
    original_manifest = json.loads((directory / "split_manifest.json").read_text())
    if generated_manifest != original_manifest:
        raise AssertionError("Current source/config does not reproduce the saved event assignments")
    summary = json.loads((directory / "data_manifest.json").read_text())
    metadata = pd.read_csv(directory / "samples.csv.gz")
    train_sensors = set(metadata.loc[metadata.split == "train", "sensor_id"])
    distribution = {}
    for split in datasets:
        subset = metadata[metadata.split == split]
        selected_events = [event for event in events if event.split == split]
        all_origins = sum(max(0, len(complete_bin_ends(event.seconds[0], event.seconds[-1])) - 30 + 1) for event in selected_events)
        counters = summary["splits"][split]
        top_sensor = subset.sensor_id.value_counts().head(1)
        distribution[split] = {
            "all_event_origins_after_30_min_history_including_gaps": all_origins,
            "input_quality_ready_origins_in_retained_segments": counters["input_quality_ready_origins"],
            "retained_segment_input_coverage_fraction": counters["input_quality_ready_origins"] / all_origins,
            "coverage_definition": "Conservative retained-segment input count divided by all in-event minute origins with elapsed 30-minute history; not operational uptime.",
            "rising_sample_fraction": float((subset.phase_at_15_minutes == "rising").mean()),
            "unseen_sensors_relative_to_train": len(set(subset.sensor_id) - train_sensors),
            "samples_from_unseen_sensors": int((~subset.sensor_id.isin(train_sensors)).sum()),
            "largest_sensor_sample_count": {str(key): int(value) for key, value in top_sensor.items()},
            "threshold_crossing_events_evaluable": counters["unique_events_reaching_threshold_from_below"],
        }
    quarantine = []
    cuts = [pd.Timestamp(value) for value in original_manifest["cut_dates_utc"]]
    embargo = pd.Timedelta(hours=config["split"]["embargo_after_cut_hours"])
    for event in events:
        if not event.quarantine:
            continue
        windows = sum(make_windows(bins, config)[1]["accepted_windows"] for _, _, bins in event_segments(event, config))
        temporal_exclusion = any(event.start < cut <= event.effective_end or cut <= event.start < cut + embargo for cut in cuts)
        provisional_split = "train" if event.start < cuts[0] else "validation" if event.start < cuts[1] else "test_temporal"
        quarantine.append({"event_id": event.event_id, "source_record": event.source_record,
                           "reasons": event.quarantine, "sample_count_if_quality_quarantine_lifted": windows,
                           "split_with_current_fixed_boundaries": "excluded_by_time" if temporal_exclusion else provisional_split,
                           "included_in_saved_data": False})
    return distribution, {
        "policy_origin": "Initial preprocessing assumption; 1 cm discrepancy cutoff is not specified in the proposal or original MD design.",
        "meaning": "Review hold, not proof of sensor error. Raw values are preserved; this audit does not restore or resplit events.",
        "events": quarantine,
        "additional_events_beyond_explicit_MD_hold": sum("configured_quarantine" not in event.quarantine for event in events if event.quarantine),
    }


def run_check(source, directory, report):
    source, directory, report = Path(source), Path(directory), Path(report)
    manifest = json.loads((directory / "data_manifest.json").read_text())
    if sha256(source) != manifest["source_sha256"]:
        raise ValueError("Source hash differs from the data-generation source")
    artifact_verification = verify_outputs(directory)
    datasets = {split: ForecastDataset(directory, split) for split in SPLITS}
    raw_streaming = raw_and_streaming_checks(source, directory, datasets)
    optimizer = optimizer_check(datasets)
    distribution, quarantine = coverage_and_quarantine(source, directory, datasets)
    document_paths = [Path("docs/forecast_pipeline.md"), Path("docs/ai_15_day_plan.md"), Path("docs/two_week_execution_plan.md")]
    document_paths += [path for path in Path.cwd().glob("*.pdf") if "기획" in unicodedata.normalize("NFC", path.name)]
    result = {
        "purpose": "LSTM data handoff and preprocessing revalidation",
        "passed": True, "source_sha256": sha256(source),
        "dataset_artifact_hashes": manifest["artifacts"],
        "data_generation_implementation_sha256": manifest["implementation_sha256"],
        "review_code_sha256": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "device": "cpu"},
        "reviewed_documents": {path.name: sha256(path) for path in document_paths},
        "specification_alignment": {
            "forecast_scope": "User update and MD: all 1..15 minutes; original proposal's 5/10/15 wording is superseded.",
            "model_input": "(B,30,1) standardized external depth only; IDs, ages, masks are metadata.",
            "model_output": "(B,15,1), LSTM(1,64,1,batch_first=True) + Linear(64,15).",
            "training": "Training-only event-balanced weighted sampler with replacement; masked MAE, Adam, gradient clipping.",
            "normalization": "X and observed Y are already standardized; loader never fits or applies another scaler.",
            "thresholds": "Evaluate 0.25 m warning and 0.30 m critical onset from the same continuous 1..15 minute forecasts.",
            "streaming": "Past-only input builder on fixed grid, with receive-time filtering and explicit WARMUP/INSUFFICIENT_DATA states.",
            "finetuning": "Original scaler can be reused; actual small-scale run-ID datasets have not been supplied.",
        },
        "artifact_verification": artifact_verification,
        "independent_csv_and_streaming_checks": raw_streaming,
        "pytorch_optimizer_check": optimizer,
        "distribution_and_coverage": distribution,
        "additional_quarantine_policy_review": quarantine,
        "remaining_requirements": [
            {"item": "Forecast performance and model selection", "status": "not_yet_evaluated",
             "detail": "Run persistence, recent-five-minute linear baseline, and full LSTM training using identical origins/masks; select on validation only. Eight optimizer steps validate integration, not accuracy."},
            {"item": "Threshold-onset evaluation", "status": "risk_test_added",
             "detail": "Use test_risk for 0.25 m warning and 0.30 m critical onset, and test_temporal for future-period generalization. Never tune on either test split; report the independent event and UTC-date counts with metrics."},
            {"item": "Unseen sensors and phase imbalance", "status": "requires_grouped_evaluation",
             "detail": "Report event-macro, sensor, seen/unseen-sensor and rising/falling/stable errors; equal-event sampling does not ensure equal-sensor or equal-phase sampling."},
            {"item": "Twelve quarantined events", "status": "requires_source_review",
             "detail": "Eleven holds were added by the initial 1 cm discrepancy rule; report their impact and resolve against source evidence before relaxing it."},
            {"item": "Small-scale finetuning and live operation", "status": "requires_new_observations",
             "required_fields": ["run_id", "measurement_time_utc", "received_time_utc", "external_depth_m", "internal_depth_m", "observation_status", "calibration_version", "pump_state", "valve_state", "control_change_time", "scenario", "preassigned_split"],
             "detail": "Collect at least 45 actual minutes per usable sequence, preferably 60–90; use separate run IDs and retain the FloodNet scaler initially. No synthetic dry history, time stretching or copied runs."},
            {"item": "Door-opening/evacuation interpretation", "status": "outside_external_depth_labels",
             "detail": "Needs independently verified physical assumptions; external 0.25/0.30 m crossings alone are not direct door-opening or evacuation ground truth."},
        ],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("export.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--report", type=Path, default=Path("analysis/lstm_readiness_review.json"))
    args = parser.parse_args()
    result = run_check(args.source, args.data, args.report)
    print(json.dumps({"passed": result["passed"], "report": str(args.report),
                      "raw_and_streaming_checks": result["independent_csv_and_streaming_checks"],
                      "optimizer_steps": result["pytorch_optimizer_check"]["training_steps"],
                      "remaining_requirements": [item["item"] for item in result["remaining_requirements"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
