"""Build a model input from past observations only, using a fixed session grid.

This pure function is shared by offline replay and a future sensor receiver. It
does not control devices, infer evacuation status, or require future labels.
"""

import numpy as np

from .preprocess import bin_observations, complete_bin_ends, validate_config


def prepare_input(observation_times_unix_seconds, depth_m, issued_at_unix_seconds,
                  grid_origin_unix_seconds, config, scaler, *, received_at_unix_seconds=None):
    validate_config(config)
    times = np.asarray(observation_times_unix_seconds, dtype=np.float64)
    depths = np.asarray(depth_m, dtype=np.float64)
    if times.ndim != 1 or times.shape != depths.shape:
        raise ValueError("Observation times and depths must be aligned one-dimensional arrays")
    received = times if received_at_unix_seconds is None else np.asarray(received_at_unix_seconds, dtype=np.float64)
    if received.shape != times.shape or not np.isfinite(received).all():
        raise ValueError("Invalid received timestamps")
    if not np.isfinite(times).all():
        raise ValueError("Invalid observation timestamp")
    now = float(issued_at_unix_seconds)
    origin = float(grid_origin_unix_seconds)
    relative_end = now - origin
    if not np.isfinite(relative_end) or relative_end < 0 or relative_end % 60 != 0 or not now.is_integer() or not origin.is_integer():
        raise ValueError("Issue time must be a completed minute boundary on the fixed session grid")
    if scaler["depth_unit"] != "m" or not np.isfinite(scaler["mean_m"]) or not np.isfinite(scaler["std_m"]) or scaler["std_m"] <= 0:
        raise ValueError("Invalid saved depth scaler")
    available = (times <= now) & (received <= now)
    times, depths = times[available], depths[available]
    if not np.isfinite(depths).all() or (depths < 0).any():
        raise ValueError("Invalid available observation depth")
    order = np.argsort(times, kind="stable")
    times, depths = times[order], depths[order]
    duplicated = np.diff(times) == 0
    if (duplicated & (np.diff(depths) != 0)).any():
        raise ValueError("Conflicting depths at one measurement timestamp")
    if len(times):
        keep = np.r_[True, ~duplicated]
        times, depths = times[keep], depths[keep]
    result = {"status": "WARMUP", "X": None, "X_observed": None, "X_age_seconds": None,
              "origin_time_unix_seconds": int(now),
              "target_bin_end_unix_seconds": int(now) + np.arange(1, 16, dtype=np.int64) * 60}
    if not len(times):
        if relative_end >= 30 * 60:
            result["status"] = "INSUFFICIENT_DATA"
        return result
    quality = config["quality"]
    breaks = np.flatnonzero(np.diff(times) > quality["split_raw_gap_gt_seconds"])
    if len(breaks):
        times, depths = times[breaks[-1] + 1:], depths[breaks[-1] + 1:]
        result["status"] = "INSUFFICIENT_DATA"
    # Do not join history across gaps or pad the start of a new segment. A
    # missing tail can be filled causally within the same configured limits.
    if times[0] > now - 30 * 60:
        return result
    if now - times[-1] > quality["max_last_observation_age_seconds"]:
        result["status"] = "INSUFFICIENT_DATA"
        return result
    relative = times - origin
    ends = complete_bin_ends(relative[0], relative_end)
    if len(ends) < 30:
        return result
    bins = bin_observations(relative, depths, ends, quality).iloc[-30:]
    observed = bins.observed.to_numpy(dtype=bool)
    inputs = bins.causal_input_depth_m.to_numpy()
    if observed.sum() < quality["min_input_observed_bins"] or not np.isfinite(inputs).all():
        result["status"] = "INSUFFICIENT_DATA"
        return result
    result.update(status="READY", X=((inputs - scaler["mean_m"]) / scaler["std_m"]).astype(np.float32)[None, :, None],
                  X_observed=observed[None, :, None],
                  X_age_seconds=bins.last_observation_age_seconds.to_numpy(dtype=np.float32)[None, :, None])
    return result
