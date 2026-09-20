"""Validate the existing four-split handoff without training or test scoring.

Run from the repository root: python -m meit_model.check_data_pipeline
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler, WeightedRandomSampler

from meit_data.torch_data import ForecastDataset, make_loader, masked_mae


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    for split in ("train", "validation", "test_temporal", "test_risk"):
        dataset = ForecastDataset(args.data_dir, split)
        require(len(dataset) > 0, f"{split}: empty dataset")
        loader = make_loader(dataset)
        expected_sampler = WeightedRandomSampler if split == "train" else SequentialSampler
        require(isinstance(loader.sampler, expected_sampler), f"{split}: unexpected sampler")

        # Check every saved sample, without rescaling or changing any arrays.
        arrays = dataset.arrays
        x, y, mask = arrays["X"], arrays["Y"], arrays["Y_mask"]
        require(x.shape == (len(dataset), 30, 1), f"{split}: bad X shape")
        require(y.shape == (len(dataset), 15, 1), f"{split}: bad Y shape")
        require(mask.shape == y.shape, f"{split}: bad mask shape")
        require(x.dtype == y.dtype == torch.float32, f"{split}: bad depth dtype")
        require(mask.dtype == torch.bool, f"{split}: bad mask dtype")
        require(torch.isfinite(x).all().item(), f"{split}: nonfinite input")
        require(torch.equal(torch.isfinite(y), mask), f"{split}: invalid target mask")

        batch = next(iter(loader))
        batch_size = len(batch["X"])
        require(batch["X"].shape == (batch_size, 30, 1), f"{split}: bad batch X")
        require(batch["Y"].shape == batch["Y_mask"].shape == (batch_size, 15, 1),
                f"{split}: bad batch targets")
        require("sample_weight" not in batch, f"{split}: loss weights exposed")
        print(f"{split}: {len(dataset):,} samples; "
              f"X={tuple(batch['X'].shape)}, Y/mask={tuple(batch['Y'].shape)}; "
              "float32/float32/bool; all arrays checked")

        if split == "train":
            # Exercise masked loss and gradients only on one training batch.
            prediction = batch["X"][:, -1:, :].repeat(1, 15, 1).requires_grad_()
            loss = masked_mae(prediction, batch["Y"], batch["Y_mask"])
            require(torch.isfinite(loss).item(), "Nonfinite training smoke-test loss")
            loss.backward()
            require(torch.isfinite(prediction.grad).all().item(), "Nonfinite gradients")
            require((prediction.grad[~batch["Y_mask"]] == 0).all().item(),
                    "Masked labels affected gradients")
            print(f"  Training smoke-test masked MAE (standardized): {loss.item():.6f}")
            print("  Masked loss/backward passed; event weights applied by sampler only.")
            evaluation = dataset.config["evaluation"]
            print("  Stored preprocessing warning/critical thresholds (m): "
                  f"{evaluation['warning_threshold_depth_m']}/"
                  f"{evaluation['threshold_depth_m']}")

    print("Existing scaler preserved; no normalization, splitting or training performed.")
    print("Both test splits checked structurally only; no test scores computed.")
    print("DATA PIPELINE CHECK PASSED")


if __name__ == "__main__":
    main()
