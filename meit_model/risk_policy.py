"""Lock a reviewed warning threshold to validation evidence and one dataset."""
from __future__ import annotations

import json
import math
from pathlib import Path

from meit_data.pipeline import sha256


POLICY_SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 3
REQUIRED_LEAD_TIME_MIN = 5
EXTERNAL_CRITICAL_DEPTH_M = .30
PRESSURE_CRITICAL_EQUIVALENT_DEPTH_M = .30
EXTERNAL_EVALUATION_SCOPE = "external_depth_forecast_crossing"
PRESSURE_POLICY_SCOPE = "outside_higher_equivalent_depth"


def threshold_contract():
    """Return the two approved 0.30 m criteria without conflating them."""
    return {
        "external_crossing": {
            "quantity": "external_depth_m",
            "critical_threshold_m": EXTERNAL_CRITICAL_DEPTH_M,
            "use": "LSTM forecast crossing evaluation",
        },
        "pressure_critical": {
            "quantity": "equivalent_depth_m",
            "critical_threshold_m": PRESSURE_CRITICAL_EQUIVALENT_DEPTH_M,
            "direction": "OUTSIDE_HIGHER",
            "use": "RiskEngine hydrostatic pressure policy",
        },
        "same_numeric_value_but_distinct_metrics": True,
    }


def read_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def _same_threshold(left, right):
    return math.isclose(float(left), float(right), rel_tol=0., abs_tol=1e-12)


def _find_threshold(rows, key, threshold):
    matches = [row for row in rows if isinstance(row, dict)
               and key in row and _same_threshold(row[key], threshold)]
    if len(matches) != 1:
        raise ValueError(f"Selected threshold {threshold:.3f} must occur once in each report")
    return matches[0]


def validate_reports(window_report, event_report, fingerprint, threshold):
    for name, report, report_type in (
            ("window calibration", window_report, "window_calibration"),
            ("event evaluation", event_report, "event_evaluation")):
        if report.get("schema_version") != REPORT_SCHEMA_VERSION:
            raise ValueError(f"Legacy {name} report is not accepted")
        if report.get("report_type") != report_type:
            raise ValueError(f"Unexpected {name} report type")
        if report.get("data_fingerprint") != fingerprint:
            raise ValueError(f"{name.capitalize()} was produced from different data")
        if report.get("evaluation_scope") != EXTERNAL_EVALUATION_SCOPE:
            raise ValueError(f"{name.capitalize()} must evaluate external-depth crossings")
        if report.get("sealed_test_used") is not False:
            raise ValueError(f"{name.capitalize()} must use development CV validation only")
        if report.get("actual_critical_depth_m") != EXTERNAL_CRITICAL_DEPTH_M:
            raise ValueError(f"{name.capitalize()} critical depth must be 0.30 m")
        if report.get("threshold_contract") != threshold_contract():
            raise ValueError(f"{name.capitalize()} threshold contract does not match the approved policy")
    if event_report.get("required_lead_time_min") != REQUIRED_LEAD_TIME_MIN:
        raise ValueError("Event evaluation must use a 5-minute required lead time")
    def evidence_hashes(report):
        return {row.get("fold"): row.get("sha256")
                for row in report.get("model_evidence", []) if isinstance(row, dict)}
    if not evidence_hashes(window_report) or evidence_hashes(window_report) != evidence_hashes(event_report):
        raise ValueError("Window and event reports must evaluate the same CV checkpoints")
    window = _find_threshold(
        window_report.get("lstm_threshold_sweep", []), "alarm_threshold_m", threshold)
    event = _find_threshold(
        event_report.get("pooled", []), "warning_threshold_m", threshold)
    event_metrics = event.get("summary", {})
    if event_metrics.get("operational_event_recall") is None:
        raise ValueError("Selected threshold has no 5-minute operational event recall")
    if event_metrics.get("required_lead_time_min") != REQUIRED_LEAD_TIME_MIN:
        raise ValueError("Selected event metrics do not use the 5-minute criterion")
    window_metrics = window.get("pooled", {})
    for metric in ("false_positive_rate", "precision", "alarm_rate"):
        if metric not in window_metrics:
            raise ValueError(f"Window calibration is missing {metric}")
    return window_metrics, event_metrics


def create_policy_decision(window_path, event_path, fingerprint, threshold, selection_note):
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 < threshold <= EXTERNAL_CRITICAL_DEPTH_M):
        raise ValueError("Selected warning threshold must be within (0, 0.30] m")
    if not isinstance(selection_note, str) or not selection_note.strip():
        raise ValueError("A non-empty human review note is required")
    window_path, event_path = Path(window_path), Path(event_path)
    window_report, event_report = read_json(window_path), read_json(event_path)
    window_metrics, event_metrics = validate_reports(
        window_report, event_report, fingerprint, float(threshold))
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "status": "LOCKED_FOR_FINAL_TRAINING",
        "data_fingerprint": fingerprint,
        "forecast_warning_depth_m": float(threshold),
        "external_critical_depth_m": EXTERNAL_CRITICAL_DEPTH_M,
        "pressure_critical_equivalent_depth_m": PRESSURE_CRITICAL_EQUIVALENT_DEPTH_M,
        "threshold_contract": threshold_contract(),
        "external_evaluation_scope": EXTERNAL_EVALUATION_SCOPE,
        "pressure_policy_scope": PRESSURE_POLICY_SCOPE,
        "required_lead_time_min": REQUIRED_LEAD_TIME_MIN,
        "selection_basis": {
            "primary_metric": "operational_event_recall_at_5_min",
            "comparison_metrics": ["false_positive_rate", "precision", "alarm_rate"],
            "review_note": selection_note.strip(),
            "selected_event_metrics": event_metrics,
            "selected_window_metrics": window_metrics,
        },
        "evidence": {
            "window_calibration": {"path": str(window_path), "sha256": sha256(window_path)},
            "event_evaluation": {"path": str(event_path), "sha256": sha256(event_path)},
        },
        "sealed_test_used": False,
    }


def load_policy_decision(decision_path, window_path, event_path, fingerprint):
    decision_path = Path(decision_path)
    decision = read_json(decision_path)
    if decision.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported risk policy decision schema")
    if decision.get("status") != "LOCKED_FOR_FINAL_TRAINING":
        raise ValueError("Risk policy decision is not locked for final training")
    if decision.get("data_fingerprint") != fingerprint:
        raise ValueError("Risk policy decision was produced from different data")
    if decision.get("sealed_test_used") is not False:
        raise ValueError("Policy selection must not use sealed test data")
    threshold = decision.get("forecast_warning_depth_m")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Risk policy has no numeric warning threshold")
    window_path, event_path = Path(window_path), Path(event_path)
    evidence = decision.get("evidence", {})
    expected = {
        "window_calibration": sha256(window_path),
        "event_evaluation": sha256(event_path),
    }
    for name, actual_hash in expected.items():
        record = evidence.get(name, {})
        if record.get("sha256") != actual_hash:
            raise ValueError(f"Risk policy {name} SHA-256 does not match supplied evidence")
    validate_reports(read_json(window_path), read_json(event_path), fingerprint, float(threshold))
    if (decision.get("required_lead_time_min") != REQUIRED_LEAD_TIME_MIN
            or decision.get("external_critical_depth_m") != EXTERNAL_CRITICAL_DEPTH_M
            or decision.get("pressure_critical_equivalent_depth_m")
            != PRESSURE_CRITICAL_EQUIVALENT_DEPTH_M
            or decision.get("threshold_contract") != threshold_contract()
            or decision.get("external_evaluation_scope") != EXTERNAL_EVALUATION_SCOPE
            or decision.get("pressure_policy_scope") != PRESSURE_POLICY_SCOPE):
        raise ValueError("Risk policy constants do not match the approved contract")
    return decision
