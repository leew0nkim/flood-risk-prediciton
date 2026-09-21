"""Read event records without using retrospective summaries as model features."""

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd


SENSOR = "Sensor ID"
START = "Flood Start Datetime (GMT)"
END = "Flood End Datetime (GMT)"
DEPTH = "Time Series Depth Values (inches)"
SECONDS = "Time Series Depth Timestamps (seconds)"
PEAK = "Maximum Flood Depth (inches)"
PEAK_TIME = "Time to Maximum Flood Depth (minutes)"
DRAIN_TIME = "Time to Drain From Peak (minutes)"
DURATION = "Total Duration (minutes)"
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"


@dataclass
class Event:
    event_id: str
    sensor_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    source_record: int
    seconds: np.ndarray
    depth_m: np.ndarray
    flags: list[str] = field(default_factory=list)
    quarantine: list[str] = field(default_factory=list)
    split: str = "excluded"
    split_reason: str = ""
    temporal_split: str = "excluded"
    event_index: int = -1

    @property
    def effective_end(self):
        return max(self.end, self.start + pd.Timedelta(seconds=float(self.seconds[-1])))


def utc_string(timestamp):
    return timestamp.isoformat().replace("+00:00", "Z")


def _number(value):
    return float(str(value).replace(",", "")) if str(value).strip() else np.nan


def read_events(source: Path, config: dict):
    """Return structurally valid events and an audit record for EVERY CSV record.

    Large summary discrepancies quarantine an event for review, not because the
    measured peak is known to be wrong. Smaller discrepancies remain flagged.
    """
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = {SENSOR, START, END, DEPTH, SECONDS, PEAK, PEAK_TIME, DRAIN_TIME, DURATION}
    if missing := required - set(raw.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    quality = config["quality"]
    conversion = config["data"]["inch_to_m"]
    explicit = {
        (item["sensor_id"], pd.Timestamp(item["flood_start_utc"])): item["reason"]
        for item in quality["quarantine_events"]
    }
    duplicates = raw.duplicated()
    key_duplicates = raw.loc[~duplicates].duplicated([SENSOR, START], keep=False)
    ambiguous_indices = set(key_duplicates.index[key_duplicates])
    events, audit = [], []
    for index, row in raw.iterrows():
        record = {
            "source_record": int(index + 2), "sensor_id": row[SENSOR],
            "event_id": "", "status": "invalid", "flags": "",
            "quarantine_reasons": "", "raw_point_count": 0,
            "retained_point_count": 0, "exact_duplicate_points_removed": 0,
        }
        flags, quarantine = [], []
        if duplicates.iloc[index]:
            record.update(status="duplicate_record", quarantine_reasons="exact_duplicate_record")
            audit.append(record)
            continue
        try:
            if not row[SENSOR].strip():
                raise ValueError("missing_sensor_id")
            start = pd.to_datetime(row[START], format=DATE_FORMAT, utc=True)
            end = pd.to_datetime(row[END], format=DATE_FORMAT, utc=True)
            if pd.isna(start) or pd.isna(end) or end < start:
                raise ValueError("invalid_event_dates")
            record["event_id"] = f"{row[SENSOR]}__{utc_string(start)}"
            seconds = np.asarray(json.loads(row[SECONDS]), dtype=np.float64)
            inches = np.asarray(json.loads(row[DEPTH]), dtype=np.float64)
            if seconds.ndim != 1 or inches.ndim != 1 or len(seconds) != len(inches):
                raise ValueError("array_length_or_dimension_mismatch")
            record["raw_point_count"] = len(seconds)
            if len(seconds) < 2:
                raise ValueError("fewer_than_two_observations")
            if not (np.isfinite(seconds).all() and np.isfinite(inches).all()):
                raise ValueError("nonfinite_array_value")
            if (seconds < 0).any() or (inches < 0).any():
                raise ValueError("negative_time_or_depth")
            if (np.diff(seconds) < 0).any():
                flags.append("observations_sorted")
                order = np.argsort(seconds, kind="stable")
                seconds, inches = seconds[order], inches[order]
            same_time = np.diff(seconds) == 0
            if (same_time & (np.diff(inches) != 0)).any():
                # Keep the original observations in the audit; do not choose one.
                raise ValueError("conflicting_depth_at_same_timestamp")
            keep = np.r_[True, ~same_time]
            removed = int((~keep).sum())
            if removed:
                flags.append("exact_duplicate_observation_removed")
            seconds, inches = seconds[keep], inches[keep]
            if len(seconds) < 2:
                raise ValueError("fewer_than_two_distinct_observations")
            depths = inches * conversion
            if index in ambiguous_indices:
                quarantine.append("duplicate_event_key")
            if (row[SENSOR], start) in explicit:
                quarantine.append("configured_quarantine")
            summaries = {}
            for name in (PEAK, PEAK_TIME, DRAIN_TIME, DURATION):
                try:
                    summaries[name] = _number(row[name])
                except ValueError:
                    summaries[name] = np.nan
                if not np.isfinite(summaries[name]):
                    flags.append(f"missing_or_invalid_summary:{name}")
            peak_diff = abs(float(inches.max()) - summaries[PEAK])
            if peak_diff > quality["summary_peak_review_tolerance_inches"]:
                flags.append("summary_peak_mismatch")
            if peak_diff * conversion > quality["summary_peak_quarantine_difference_m"]:
                quarantine.append("large_summary_peak_difference")
            if summaries[PEAK_TIME] > summaries[DURATION]:
                flags.append("peak_time_exceeds_duration")
            span_seconds = float(seconds[-1] - seconds[0])
            if abs(span_seconds - summaries[DURATION] * 60) > 1:
                flags.append("summary_duration_mismatch")
            if seconds[0] > 0:
                flags.append("observations_start_after_event_start")
            series_end = start + pd.Timedelta(seconds=float(seconds[-1]))
            if abs((series_end - end).total_seconds()) > 1:
                flags.append("summary_end_timestamp_mismatch")
            event = Event(record["event_id"], row[SENSOR], start, end,
                          int(index + 2), seconds, depths, flags, quarantine)
            events.append(event)
            record.update(
                status="quarantined" if quarantine else "eligible",
                raw_point_count=len(keep), retained_point_count=len(seconds),
                exact_duplicate_points_removed=removed,
                start_utc=utc_string(start), end_utc=utc_string(end),
                first_observation_utc=utc_string(start + pd.Timedelta(seconds=float(seconds[0]))),
                last_observation_utc=utc_string(series_end),
                series_peak_m=float(depths.max()), summary_peak_m=summaries[PEAK] * conversion,
                peak_difference_m=peak_diff * conversion,
                span_minutes=span_seconds / 60,
                gaps_gt_limit=int((np.diff(seconds) > quality["split_raw_gap_gt_seconds"]).sum()),
            )
        except (ValueError, TypeError, OverflowError) as exc:
            quarantine.append(str(exc))
        record["flags"] = ";".join(flags)
        record["quarantine_reasons"] = ";".join(quarantine)
        audit.append(record)
    events.sort(key=lambda e: (e.start, e.event_id, e.source_record))
    for i, event in enumerate(events):
        event.event_index = i
    info = {
        "records": len(raw), "columns": len(raw.columns),
        "sensors": int(raw[SENSOR].nunique()),
        "blank_cells": {c: int(raw[c].str.strip().eq("").sum()) for c in raw.columns},
        "numeric_comma_cells": {c: int(raw[c].str.contains(",").sum()) for c in raw.columns[4:11]},
    }
    return events, pd.DataFrame(audit), info


def assign_splits(events: list[Event], config: dict):
    """Choose UTC day boundaries nearest requested cumulative event counts.

    Full original events, including every sensor on the same date, share a
    partition. A temporal buffer is not a guarantee of storm independence.
    """
    eligible = [event for event in events if not event.quarantine]
    if not eligible:
        raise ValueError("No eligible events")
    starts = pd.Series([event.start for event in eligible]).sort_values()
    dates = sorted(set(starts.dt.normalize()))
    if len(dates) < 3:
        raise ValueError("At least three UTC start dates are needed for chronological splits")
    ratios = config["split"]["target_ratios"]
    targets = np.cumsum(ratios)[:2] * len(starts)
    candidates = dates[1:]
    counts = np.array([int((starts < date).sum()) for date in candidates])
    # Jointly choose distinct boundaries, preserving temporal order.
    pairs = ((abs(counts[i] - targets[0]) + abs(counts[j] - targets[1]), i, j)
             for i in range(len(candidates) - 1) for j in range(i + 1, len(candidates)))
    _, i, j = min(pairs)
    cuts = [candidates[i], candidates[j]]
    embargo = pd.Timedelta(hours=config["split"]["embargo_after_cut_hours"])
    for event in events:
        event.split, event.split_reason = "excluded", ""
        if event.quarantine:
            event.split_reason = "quarantined"
        elif any(event.start < cut <= event.effective_end for cut in cuts):
            event.split_reason = "boundary_crossing_event"
        elif any(cut <= event.start < cut + embargo for cut in cuts):
            event.split_reason = "embargo"
        else:
            event.split = "train" if event.start < cuts[0] else "validation" if event.start < cuts[1] else "test_temporal"
        event.temporal_split = event.split
    return {
        "method": config["split"]["method"], "target_ratios": ratios,
        "cut_dates_utc": [utc_string(cut) for cut in cuts],
        "embargo_after_cut_hours": config["split"]["embargo_after_cut_hours"],
        "eligible_events_before_temporal_exclusions": len(eligible),
        "storm_grouping": "No storm IDs available; same-date grouping plus time embargo only",
        "events": [{
            "event_index": e.event_index, "event_id": e.event_id, "sensor_id": e.sensor_id,
            "source_record": e.source_record, "start_utc": utc_string(e.start),
            "effective_end_utc": utc_string(e.effective_end), "split": e.split,
            "temporal_split": e.temporal_split,
            "exclusion_reason": e.split_reason, "quarantine_reasons": e.quarantine,
        } for e in events],
    }
