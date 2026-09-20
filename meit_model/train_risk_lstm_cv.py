from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from meit_model.cv import (
    load_development_data,
    load_scaler,
)

from meit_model.lstm import (
    LSTMForecaster,
)

from meit_model.risk_loss import (
    risk_aware_masked_mae,
)

from meit_model.train_lstm_cv import (
    CVDataset,
    evaluate_mae_m,
    make_event_balanced_loader,
    make_validation_loader,
    set_seed,
)


def train_one_epoch_risk(
    model,
    loader,
    optimizer,
    scaler,
    device,
    grad_clip,
    critical_depth_m,
    underprediction_weight,
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

        loss = risk_aware_masked_mae(
            prediction=prediction,
            target=Y,
            observed=Y_mask,
            scaler=scaler,
            critical_depth_m=(
                critical_depth_m
            ),
            underprediction_weight=(
                underprediction_weight
            ),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
        )

        optimizer.step()

        loss_sum += loss.item()
        batch_count += 1

    return (
        loss_sum
        / batch_count
    )


def train_fold(
    fold_number,
    train_indices,
    val_indices,
    data,
    scaler,
    output_dir,
    device,
    config,
):

    print(
        "\n"
        "================================"
    )

    print(
        f"RISK-AWARE FOLD {fold_number}"
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
            batch_size=config[
                "batch_size"
            ],
            seed=config["seed"],
        )
    )

    val_loader = (
        make_validation_loader(
            val_dataset,
            batch_size=config[
                "batch_size"
            ],
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
        lr=config[
            "learning_rate"
        ],
    )

    best_val_mae_m = float(
        "inf"
    )

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
        config[
            "max_epochs"
        ] + 1,
    ):

        train_loss = (
            train_one_epoch_risk(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                grad_clip=config[
                    "gradient_clip"
                ],
                critical_depth_m=config[
                    "critical_depth_m"
                ],
                underprediction_weight=config[
                    "underprediction_weight"
                ],
            )
        )

        # --------------------------------------------
        # checkpoint 선택 기준은 기존과 동일!
        #
        # validation overall MAE in metres
        # --------------------------------------------

        val_mae_m = evaluate_mae_m(
            model=model,
            loader=val_loader,
            scaler=scaler,
            device=device,
        )

        val_mae_cm = (
            val_mae_m
            * 100
        )

        print(
            f"Epoch {epoch:02d} | "
            f"risk loss="
            f"{train_loss:.6f} | "
            f"val MAE="
            f"{val_mae_cm:.4f} cm"
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_risk_loss":
                    train_loss,

                "val_mae_m":
                    val_mae_m,

                "val_mae_cm":
                    val_mae_cm,
            }
        )

        if (
            val_mae_m
            < best_val_mae_m
        ):

            best_val_mae_m = (
                val_mae_m
            )

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
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

                    "loss":
                        "risk_aware_asymmetric_mae",
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
            best_val_mae_m
            * 100,
    }

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best val MAE:",
        f"{best_val_mae_m * 100:.4f} cm"
    )

    return result


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
            "artifacts/risk_lstm_cv"
        ),
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    config = {
        "input_size":
            1,

        "hidden_size":
            64,

        "horizon":
            15,

        "batch_size":
            128,

        "learning_rate":
            0.001,

        "max_epochs":
            50,

        "patience":
            8,

        "gradient_clip":
            1.0,

        "seed":
            42,

        # ========================================
        # Risk-aware 설정
        # ========================================

        "critical_depth_m":
            0.30,

        "underprediction_weight":
            2.0,
    }

    set_seed(
        config["seed"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    print(
        "Critical depth:",
        config[
            "critical_depth_m"
        ],
        "m",
    )

    print(
        "Underprediction weight:",
        config[
            "underprediction_weight"
        ],
    )

    data = load_development_data(
        args.data_dir
    )

    scaler = load_scaler(
        args.data_dir
    )

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

    for fold_number in (
        fold_numbers
    ):

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
                fold[
                    "train_indices"
                ]
            )

            val_indices = (
                fold[
                    "val_indices"
                ]
            )

        result = train_fold(
            fold_number=fold_number,
            train_indices=train_indices,
            val_indices=val_indices,
            data=data,
            scaler=scaler,
            output_dir=args.output_dir,
            device=device,
            config=config,
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
