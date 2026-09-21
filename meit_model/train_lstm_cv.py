from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from meit_model.fold_scaling import FoldScalerSource, rescale_data, indices_digest
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

from meit_data.torch_data import (
    depth_to_m,
    horizon_errors,
    masked_mae,
)

from meit_model.cv import (
    load_development_data,
    load_scaler,
)

from meit_model.lstm import (
    LSTMForecaster,
)


# ============================================================
# 재현성
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CV용 Dataset
# ============================================================

class CVDataset(Dataset):

    def __init__(
        self,
        data,
        indices: np.ndarray,
    ):
        self.data = data
        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]

        return {
            "X": torch.from_numpy(
                np.ascontiguousarray(
                    self.data.X[index]
                )
            ).float(),

            "Y": torch.from_numpy(
                np.ascontiguousarray(
                    self.data.Y[index]
                )
            ).float(),

            "Y_mask": torch.from_numpy(
                np.ascontiguousarray(
                    self.data.Y_mask[index]
                )
            ).bool(),

            "event_index": torch.tensor(
                self.data.event_index[index],
                dtype=torch.long,
            ),
        }


# ============================================================
# Event-balanced training sampler
# ============================================================

def make_event_balanced_loader(
    dataset: CVDataset,
    batch_size: int,
    seed: int,
):

    event_ids = (
        dataset.data.event_index[
            dataset.indices
        ]
    )

    unique_events, counts = np.unique(
        event_ids,
        return_counts=True,
    )

    count_map = dict(
        zip(
            unique_events.tolist(),
            counts.tolist(),
        )
    )

    # 각 event의 전체 sampling weight가 동일해지도록
    # sample 하나의 weight = 1 / 해당 event의 sample 수
    weights = np.array(
        [
            1.0 / count_map[event_id]
            for event_id in event_ids
        ],
        dtype=np.float64,
    )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def make_validation_loader(
    dataset: CVDataset,
    batch_size: int,
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


# ============================================================
# Training epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip: float,
):

    model.train()

    loss_sum = 0.0
    batch_count = 0

    for batch in loader:

        X = batch["X"].to(device)
        Y = batch["Y"].to(device)
        Y_mask = (
            batch["Y_mask"]
            .to(device)
        )

        optimizer.zero_grad()

        prediction = model(X)

        loss = masked_mae(
            prediction,
            Y,
            Y_mask,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
        )

        optimizer.step()

        loss_sum += loss.item()
        batch_count += 1

    return loss_sum / batch_count


# ============================================================
# Validation - m 단위 MAE
# ============================================================

@torch.no_grad()
def evaluate_mae_m(
    model,
    loader,
    scaler,
    device,
):

    model.eval()

    absolute_error_sum = torch.zeros(
        15,
        dtype=torch.float64,
    )

    observed_count = torch.zeros(
        15,
        dtype=torch.int64,
    )

    for batch in loader:

        X = batch["X"].to(device)
        Y = batch["Y"].to(device)
        Y_mask = (
            batch["Y_mask"]
            .to(device)
        )

        prediction = model(X)

        prediction_m = depth_to_m(
            prediction,
            scaler,
            clip_negative=True,
        )

        target_m = depth_to_m(
            Y,
            scaler,
            clip_negative=False,
        )

        stats = horizon_errors(
            prediction_m,
            target_m,
            Y_mask,
        )

        absolute_error_sum += (
            stats[
                "absolute_error_sum_m"
            ]
            .cpu()
        )

        observed_count += (
            stats[
                "observed_count"
            ]
            .cpu()
        )

    total_count = (
        observed_count.sum()
    )

    overall_mae_m = (
        absolute_error_sum.sum()
        / total_count
    )

    return float(overall_mae_m)


# ============================================================
# Fold 하나 학습
# ============================================================

def train_fold(
    fold_number,
    train_indices,
    val_indices,
    data,
    scaler,
    output_dir,
    device,
    config,
    provenance,
):

    set_seed(config["seed"])
    if (output_dir / f"fold_{fold_number}" / "best_model.pt").exists():
        raise FileExistsError("Checkpoint already exists; choose a new --output-dir")

    print(
        "\n"
        "================================"
    )

    print(
        f"FOLD {fold_number}"
    )

    print(
        "================================"
    )

    train_dataset = CVDataset(
        data,
        train_indices,
    )

    val_dataset = CVDataset(
        data,
        val_indices,
    )

    train_loader = (
        make_event_balanced_loader(
            train_dataset,
            batch_size=config["batch_size"],
            seed=config["seed"],
        )
    )

    val_loader = (
        make_validation_loader(
            val_dataset,
            batch_size=config["batch_size"],
        )
    )

    model = LSTMForecaster(
        input_size=1,
        hidden_size=config[
            "hidden_size"
        ],
        num_layers=1,
        horizon=15,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    best_val_mae_m = float("inf")
    best_epoch = None
    patience_counter = 0

    history = []

    fold_dir = (
        output_dir
        / f"fold_{fold_number}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        config["max_epochs"] + 1,
    ):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=config[
                "gradient_clip"
            ],
        )

        val_mae_m = evaluate_mae_m(
            model=model,
            loader=val_loader,
            scaler=scaler,
            device=device,
        )

        val_mae_cm = (
            val_mae_m * 100
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train standardized MAE="
            f"{train_loss:.6f} | "
            f"val MAE="
            f"{val_mae_cm:.4f} cm"
        )

        history.append(
            {
                "epoch": epoch,
                "train_standardized_mae":
                    train_loss,
                "val_mae_m":
                    val_mae_m,
                "val_mae_cm":
                    val_mae_cm,
            }
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_mae_m < best_val_mae_m:

            best_val_mae_m = (
                val_mae_m
            )

            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "normalization_version": 2,
                    "scaler": scaler,
                    **provenance,
                    "fold":
                        fold_number,

                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_mae_m":
                        val_mae_m,

                    "config":
                        config,
                },

                fold_dir
                / "best_model.pt",
            )

        else:

            patience_counter += 1

            if (
                patience_counter
                >= config[
                    "patience"
                ]
            ):

                print(
                    "Early stopping "
                    f"at epoch {epoch}"
                )

                break

    with open(
        fold_dir
        / "history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            ensure_ascii=False,
            indent=2,
        )

    result = {
        "fold":
            fold_number,

        "best_epoch":
            best_epoch,

        "best_val_mae_m":
            best_val_mae_m,

        "best_val_mae_cm":
            best_val_mae_m * 100,
    }

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best val MAE: "
        f"{best_val_mae_m * 100:.4f} cm"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "data/processed"
        ),
    )

    parser.add_argument(
        "--cv-dir",
        type=Path,
        default=Path(
            "artifacts/cv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/lstm_cv_fold_scaled"
        ),
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
    )

    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.fold not in (None, 1, 2, 3, 4):
        parser.error("max-epochs must be positive; fold must be 1..4")

    config = {
        "input_size": 1,
        "hidden_size": 64,
        "horizon": 15,

        "batch_size": 128,
        "learning_rate": 0.001,

        "max_epochs": 50,
        "patience": 8,
        "gradient_clip": 1.0,

        "seed": 42,
    }

    config["max_epochs"] = args.max_epochs
    config["seed"] = args.seed
    config["normalization_version"] = 2
    set_seed(config["seed"])

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    data = load_development_data(
        args.data_dir
    )

    scaler = load_scaler(
        args.data_dir
    )

    scaling_source = FoldScalerSource(args.data_dir)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.fold is None:
        fold_numbers = [
            1,
            2,
            3,
            4,
        ]
    else:
        fold_numbers = [
            args.fold
        ]

    results = []

    for fold_number in fold_numbers:

        fold_path = (
            args.cv_dir
            / f"fold_{fold_number}.npz"
        )

        if not fold_path.exists():
            raise FileNotFoundError(
                fold_path
            )

        with np.load(
            fold_path,
            allow_pickle=False,
        ) as fold:

            train_indices = (
                fold["train_indices"]
            )

            val_indices = (
                fold["val_indices"]
            )

        fold_scaler = scaling_source.fit(train_indices)
        fold_data = rescale_data(data, scaler, fold_scaler)
        provenance = {
            "data_fingerprint": scaling_source.fingerprint,
            "train_indices_sha256": indices_digest(train_indices),
            "val_indices_sha256": indices_digest(val_indices),
        }

        result = train_fold(
            fold_number=fold_number,
            train_indices=train_indices,
            val_indices=val_indices,
            data=fold_data,
            scaler=fold_scaler,
            output_dir=args.output_dir,
            device=device,
            config=config,
            provenance=provenance,
        )

        results.append(
            result
        )

    with open(
        args.output_dir
        / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
