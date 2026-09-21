"""Fold-local scaling without modifying the supplied preprocessing artifacts."""

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from meit_data.pipeline import sha256
from meit_data.preprocess import fit_scaler


def indices_digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def data_fingerprint(directory):
    directory = Path(directory)
    return {name: sha256(directory / name) for name in
            ("train.npz", "validation.npz", "scaler.json", "minute_bins.csv.gz")}


class FoldScalerSource:
    """Use raw observed bins covered by training inputs, counting each bin once."""

    def __init__(self, directory):
        directory = Path(directory)
        parts = []
        for split in ("train", "validation"):
            with np.load(directory / f"{split}.npz", allow_pickle=False) as archive:
                parts.append({key: archive[key] for key in
                              ("segment_index", "input_start_bin_index")})
        self.segments = np.concatenate([p["segment_index"] for p in parts])
        self.starts = np.concatenate([p["input_start_bin_index"] for p in parts])
        bins = pd.read_csv(directory / "minute_bins.csv.gz")
        self.bins = {int(key): group.reset_index(drop=True)
                     for key, group in bins.groupby("segment_index", sort=False)}
        self.fingerprint = data_fingerprint(directory)

    def fit(self, train_indices):
        indices = np.asarray(train_indices, dtype=np.int64)
        if (indices.ndim != 1 or not len(indices) or indices.min() < 0
                or indices.max() >= len(self.starts)
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("Invalid fold training indices")
        values = []
        segments = self.segments[indices]
        for segment in np.unique(segments):
            bins = self.bins[int(segment)]
            starts = self.starts[indices[segments == segment]]
            if starts.min() < 0 or (starts + 30).max() > len(bins):
                raise ValueError("Training input extends beyond minute table")
            changes = np.zeros(len(bins) + 1, dtype=np.int64)
            np.add.at(changes, starts, 1)
            np.add.at(changes, starts + 30, -1)
            selected = (changes[:-1].cumsum() > 0) & bins.observed.to_numpy(bool)
            values.append(bins.observed_depth_m.to_numpy(np.float64)[selected])
        scaler = fit_scaler(np.concatenate(values))
        scaler["scope"] = "cv_fold_training_inputs_only"
        return scaler


def rescale_data(data, source_scaler, target_scaler):
    """Invert the supplied transform, then apply the training-fold transform."""
    for scaler in (source_scaler, target_scaler):
        if not np.isfinite(scaler["mean_m"]) or not np.isfinite(scaler["std_m"]) or scaler["std_m"] <= 0:
            raise ValueError("Invalid scaler")

    def transform(array):
        metres = array.astype(np.float64) * source_scaler["std_m"] + source_scaler["mean_m"]
        return ((metres - target_scaler["mean_m"]) / target_scaler["std_m"]).astype(np.float32)

    return replace(data, X=transform(data.X), Y=transform(data.Y))


def checkpoint_data(data, source_scaler, checkpoint, fingerprint, val_indices):
    """Reject legacy/stale checkpoints instead of silently using a wrong scaler."""
    if checkpoint.get("normalization_version") != 2 or "scaler" not in checkpoint:
        raise ValueError("Checkpoint has no fold-local scaler. Retrain with the corrected CV trainer.")
    if checkpoint.get("data_fingerprint") != fingerprint:
        raise ValueError("Checkpoint was trained on different preprocessing artifacts")
    if checkpoint.get("val_indices_sha256") != indices_digest(val_indices):
        raise ValueError("Checkpoint validation indices do not match the selected CV fold")
    scaler = checkpoint["scaler"]
    return rescale_data(data, source_scaler, scaler), scaler
