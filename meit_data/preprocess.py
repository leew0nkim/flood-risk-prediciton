"""Causal minute bins, supervised windows, and train-only scaling.

Every complete bin is (left, right]. Values at a boundary enter exactly one
bin. Bins must fit inside the observed span of a single continuous segment.
The observation at/before a bin's left edge may seed a bounded forward fill,
but is never labeled as an observation INSIDE that bin.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .io import Event


def validate_config(config):
    fixed = {
        ("data", "depth_unit"): "m", ("data", "source_depth_unit"): "inches",
        ("data", "inch_to_m"): .0254, ("data", "time_zone"): "UTC",
        ("data", "bin_seconds"): 60, ("data", "bin_closed"): "right",
        ("data", "bin_representative"): "last_valid_observation",
        ("data", "offline_grid_origin"): "event_start",
        ("window", "input_bins"): 30, ("window", "output_bins"): 15,
        ("window", "input_features"): ["depth_m"],
        ("window", "horizon_minutes"): list(range(1, 16)),
        ("quality", "input_fill"): "forward_fill", ("quality", "target_fill"): "none",
        ("quality", "loss_and_metric_mask"): "observed_and_finite",
        ("split", "method"): "chronological_by_whole_event_and_date",
        ("split", "boundary_crossing_event_policy"): "exclude",
        ("scaling", "method"): "standard",
        ("scaling", "fit_on"): "unique_observed_training_input_bins",
        ("scaling", "share_depth_transform_for_input_and_target"): True,
        ("training", "window_sampling"): "equal_total_weight_per_event",
    }
    for (section, key), expected in fixed.items():
        if config[section][key] != expected:
            raise ValueError(f"Unsupported {section}.{key}: expected {expected!r}")
    window, quality = config["window"], config["quality"]
    for key in ("input_bins", "output_bins", "stride_bins"):
        if not isinstance(window[key], int) or window[key] < 1:
            raise ValueError(f"Invalid {key}")
    for key, maximum in (("min_input_observed_bins", window["input_bins"]),
                         ("min_target_observed_bins", window["output_bins"])):
        if not isinstance(quality[key], int) or not 1 <= quality[key] <= maximum:
            raise ValueError(f"Invalid {key}")
    if not isinstance(quality["max_consecutive_missing_input_bins"], int) or quality["max_consecutive_missing_input_bins"] < 0:
        raise ValueError("Invalid maximum consecutive missing bins")
    for key in ("split_raw_gap_gt_seconds", "max_last_observation_age_seconds",
                "summary_peak_review_tolerance_inches", "summary_peak_quarantine_difference_m"):
        if not np.isfinite(quality[key]) or quality[key] <= 0:
            raise ValueError(f"Invalid {key}")
    ratios = np.asarray(config["split"]["target_ratios"])
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1):
        raise ValueError("Split ratios must contain three positive values summing to one")
    if not np.isfinite(config["split"]["embargo_after_cut_hours"]) or config["split"]["embargo_after_cut_hours"] < 0:
        raise ValueError("Invalid embargo duration")


def complete_bin_ends(first_second, last_second, width=60):
    first_end = (int(np.ceil(first_second / width)) + 1) * width
    last_end = int(np.floor(last_second / width)) * width
    return np.arange(first_end, last_end + 1, width, dtype=np.int64)


def bin_observations(seconds, depth_m, ends, quality, width=60):
    """Shared causal bin operation: only observations at/before each end count.

    Callers supply one gap-free segment and fixed bin ends. Future values in
    the supplied arrays cannot affect existing input bins.
    """
    if len(seconds) == 0 or len(seconds) != len(depth_m) or (np.diff(seconds) <= 0).any():
        raise ValueError("Expected nonempty, strictly increasing observations")
    last_indices = np.searchsorted(seconds, ends, side="right") - 1
    have_past = last_indices >= 0
    safe_indices = np.maximum(last_indices, 0)
    last_seconds = np.where(have_past, seconds[safe_indices], np.nan)
    age = ends - last_seconds
    observed = have_past & (last_seconds > ends - width)
    observed_depth = np.where(observed, depth_m[safe_indices], np.nan)
    missing_streak = np.zeros(len(ends), dtype=np.int32)
    for i in range(len(ends)):
        if not observed[i]:
            missing_streak[i] = 1 + (missing_streak[i - 1] if i else 0)
    can_fill = (have_past & (missing_streak <= quality["max_consecutive_missing_input_bins"])
                & (age <= quality["max_last_observation_age_seconds"]))
    inputs = np.where(observed | can_fill, depth_m[safe_indices], np.nan)
    return pd.DataFrame({
        "bin_end_relative_seconds": ends, "observed_depth_m": observed_depth,
        "observed": observed, "last_observation_relative_seconds": last_seconds,
        "last_observation_age_seconds": age, "causal_input_depth_m": inputs,
    })


def event_segments(event: Event, config: dict):
    """Yield segment number, raw point count, and minute table, even if empty."""
    quality = config["quality"]
    boundaries = np.r_[0, np.flatnonzero(np.diff(event.seconds) > quality["split_raw_gap_gt_seconds"]) + 1,
                       len(event.seconds)]
    for number, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        seconds, depth = event.seconds[left:right], event.depth_m[left:right]
        ends = complete_bin_ends(seconds[0], seconds[-1], config["data"]["bin_seconds"])
        bins = bin_observations(seconds, depth, ends, quality, config["data"]["bin_seconds"])
        bins["bin_end_unix_seconds"] = int(event.start.timestamp()) + ends
        yield number, right - left, bins


def empty_windows(config):
    x, y = config["window"]["input_bins"], config["window"]["output_bins"]
    return {
        "X": np.empty((0, x, 1), dtype=np.float64),
        "Y": np.empty((0, y, 1), dtype=np.float64),
        "X_observed": np.empty((0, x, 1), dtype=bool),
        "X_age_seconds": np.empty((0, x, 1), dtype=np.float32),
        "Y_mask": np.empty((0, y, 1), dtype=bool),
        "origin_time_unix_seconds": np.empty(0, dtype=np.int64),
        "input_start_bin_index": np.empty(0, dtype=np.int64),
    }


def make_windows(bins, config):
    lookback, horizon = config["window"]["input_bins"], config["window"]["output_bins"]
    stride = config["window"]["stride_bins"]
    quality = config["quality"]
    stats = {"minute_bins": len(bins), "observed_minute_bins": int(bins.observed.sum()),
             "history_origins_in_segments": max(0, len(bins) - lookback + 1),
             "input_quality_ready_origins": 0, "candidate_windows": 0, "accepted_windows": 0,
             "insufficient_input_observations": 0, "unfilled_input": 0,
             "insufficient_target_observations": 0}
    if len(bins) < lookback:
        return empty_windows(config), stats
    observed = bins.observed.to_numpy(dtype=bool)
    input_windows = sliding_window_view(bins.causal_input_depth_m.to_numpy(), lookback)
    input_observed = sliding_window_view(observed, lookback)
    enough_input = input_observed.sum(axis=1) >= quality["min_input_observed_bins"]
    finite_input = np.isfinite(input_windows).all(axis=1)
    stats["input_quality_ready_origins"] = int((enough_input & finite_input).sum())
    if len(bins) < lookback + horizon:
        return empty_windows(config), stats
    indices = np.arange(0, len(bins) - lookback - horizon + 1, stride)
    y = sliding_window_view(bins.observed_depth_m.to_numpy()[lookback:], horizon)[indices]
    mask = sliding_window_view(observed[lookback:], horizon)[indices]
    enough_target = mask.sum(axis=1) >= quality["min_target_observed_bins"]
    good_input, finite = enough_input[indices], finite_input[indices]
    accept = good_input & finite & enough_target
    stats.update(candidate_windows=len(indices), accepted_windows=int(accept.sum()),
                 insufficient_input_observations=int((~good_input).sum()),
                 unfilled_input=int((good_input & ~finite).sum()),
                 insufficient_target_observations=int((good_input & finite & ~enough_target).sum()))
    selected = indices[accept]
    return {
        "X": input_windows[selected, :, None].copy(), "Y": y[accept, :, None].copy(),
        "X_observed": input_observed[selected, :, None].copy(),
        "X_age_seconds": sliding_window_view(bins.last_observation_age_seconds.to_numpy(), lookback)[selected, :, None].astype(np.float32),
        "Y_mask": mask[accept, :, None].copy(),
        "origin_time_unix_seconds": bins.bin_end_unix_seconds.to_numpy()[selected + lookback - 1],
        "input_start_bin_index": selected,
    }, stats


def scaler_bin_mask(bins, windows, lookback):
    """Count unique observed input bins, not repeated sliding-window entries."""
    changes = np.zeros(len(bins) + 1, dtype=np.int64)
    starts = windows["input_start_bin_index"]
    np.add.at(changes, starts, 1)
    np.add.at(changes, starts + lookback, -1)
    return (changes[:-1].cumsum() > 0) & bins.observed.to_numpy(dtype=bool)


def fit_scaler(unique_observed_train_values):
    values = np.asarray(unique_observed_train_values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Scaler requires finite, unique, observed training input bins")
    mean, std = float(values.mean()), float(values.std(ddof=0))
    if std <= 0:
        raise ValueError("Training depth standard deviation must be positive")
    return {"method": "standard", "depth_unit": "m", "mean_m": mean, "std_m": std,
            "ddof": 0, "fit_on": "unique_observed_training_input_bins",
            "unique_observed_bin_count": int(values.size), "shared_X_Y_transform": True}


def normalize_windows(windows, scaler):
    result = dict(windows)
    result["X"] = ((windows["X"] - scaler["mean_m"]) / scaler["std_m"]).astype(np.float32)
    result["Y"] = ((windows["Y"] - scaler["mean_m"]) / scaler["std_m"]).astype(np.float32)
    result["Y"][~result["Y_mask"]] = np.nan
    return result


def event_sampling_weights(event_indices):
    _, inverse, counts = np.unique(event_indices, return_inverse=True, return_counts=True)
    return (1.0 / counts[inverse]).astype(np.float64)
