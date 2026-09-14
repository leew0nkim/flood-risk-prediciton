"""PyTorch handoff for already-standardized FloodNet arrays.

    train = ForecastDataset("data/processed", "train")
    loader = make_loader(train)  # equal event weight, replacement=True
    batch = next(iter(loader))
    prediction = model(batch["X"])  # (B, 15, 1)
    loss = masked_mae(prediction, batch["Y"], batch["Y_mask"])
    loss.backward()

Do not fit another scaler, apply ReLU to standardized predictions, or apply
sample weights again in the loss. For reported metres, use depth_to_m with
clip_negative=True on predictions and False on targets.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .pipeline import sha256
from .preprocess import validate_config


class ForecastDataset(Dataset):
    """Load one verified split. No rescaling, shuffling or dropping labels here."""

    def __init__(self, directory, split):
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        directory = Path(directory)
        self.split = split
        manifest = json.loads((directory / "data_manifest.json").read_text())
        for name in (f"{split}.npz", "config.json", "scaler.json", "split_manifest.json"):
            if sha256(directory / name) != manifest["artifacts"][name]["sha256"]:
                raise ValueError(f"Artifact hash mismatch: {name}")
        if manifest["contract"]["saved_depth_arrays"] != "standardized_float32":
            raise ValueError("Expected arrays already standardized with the supplied scaler")
        self.config = json.loads((directory / "config.json").read_text())
        self.scaler = json.loads((directory / "scaler.json").read_text())
        validate_config(self.config)
        with np.load(directory / f"{split}.npz", allow_pickle=False) as archive:
            self.arrays = {key: torch.from_numpy(np.ascontiguousarray(archive[key])) for key in archive.files}
        if self.arrays["X"].dtype != torch.float32 or self.arrays["Y"].dtype != torch.float32:
            raise ValueError("Model depth arrays must be float32")
        if not torch.equal(torch.isfinite(self.arrays["Y"]), self.arrays["Y_mask"]):
            raise ValueError("Target observations and target mask disagree")
        assignments = json.loads((directory / "split_manifest.json").read_text())["events"]
        self.events = {event["event_index"]: event for event in assignments if event["split"] == split}
        if not set(self.arrays["event_index"].unique().tolist()).issubset(self.events):
            raise ValueError("Array event identifiers do not belong to the requested split")
        self.sampling_weights = self.arrays["sample_weight"]
        self.horizon_offsets_seconds = torch.arange(1, 16, dtype=torch.int64) * 60

    def __len__(self):
        return len(self.arrays["X"])

    def __getitem__(self, index):
        # Expose weights only through the sampler, avoiding accidental double
        # weighting in the training loss. IDs stay available for grouped metrics.
        batch = {key: value[index] for key, value in self.arrays.items() if key != "sample_weight"}
        batch["target_bin_end_unix_seconds"] = batch["origin_time_unix_seconds"] + self.horizon_offsets_seconds
        return batch


def make_loader(dataset, batch_size=None, seed=None):
    """One epoch has len(train) draws; validation/test are complete and ordered.

    num_workers=0 works in scripts and notebooks without platform-specific
    multiprocessing setup. An independent generator makes sampling repeatable.
    """
    batch_size = batch_size or dataset.config["training"]["batch_size"]
    seed = dataset.config["training"]["seed"] if seed is None else seed
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if dataset.split == "train":
        sampler = WeightedRandomSampler(dataset.sampling_weights, num_samples=len(dataset),
                                        replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                      drop_last=False, num_workers=0, generator=generator)


def masked_mae(prediction, target, observed):
    """Select observed labels BEFORE arithmetic so NaN targets cannot poison gradients.

    An empty valid set is an error; a caller handling arbitrary batches can catch
    it and skip that batch. Every exported training sample has at least 14 labels.
    """
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError("Prediction, target and mask shapes must match")
    if observed.dtype != torch.bool:
        raise TypeError("Observation mask must have boolean dtype")
    valid = observed & torch.isfinite(target)
    if not valid.any():
        raise ValueError("No observed finite labels in this batch")
    if not torch.isfinite(prediction[valid]).all():
        raise ValueError("Nonfinite prediction at an observed target")
    return torch.abs(prediction[valid] - target[valid]).mean()


def depth_to_m(standardized, scaler, *, clip_negative=False):
    depth = standardized * scaler["std_m"] + scaler["mean_m"]
    return torch.clamp_min(depth, 0) if clip_negative else depth


def horizon_errors(prediction_m, target_m, observed):
    """Return additive statistics, so unequal batches do not bias epoch metrics."""
    if prediction_m.shape != target_m.shape or observed.shape != target_m.shape:
        raise ValueError("Metric arrays must have matching shapes")
    valid = observed & torch.isfinite(target_m)
    if not torch.isfinite(prediction_m[valid]).all():
        raise ValueError("Nonfinite prediction at an observed target")
    # Mask BEFORE subtraction. Zero here is only a summation identity; counts
    # still exclude the missing observations and no label is manufactured.
    safe_prediction = torch.where(valid, prediction_m, torch.zeros_like(prediction_m))
    safe_target = torch.where(valid, target_m, torch.zeros_like(target_m))
    error = (safe_prediction - safe_target).double()
    return {"absolute_error_sum_m": error.abs().sum(dim=(0, 2)),
            "squared_error_sum_m2": error.square().sum(dim=(0, 2)),
            "observed_count": valid.sum(dim=(0, 2))}
