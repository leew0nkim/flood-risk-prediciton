from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from meit_data.pipeline import sha256
from meit_model.fold_scaling import FoldScalerSource, rescale_data, indices_digest

from meit_model.cv import (
    CV_SCHEMA_VERSION,
    DEFAULT_CRITICAL_DEPTH_M,
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

                    "loss":
                        "risk_aware_asymmetric_mae_valid_mean_v2",
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
            "artifacts/risk_lstm_cv_fold_scaled"
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
            args.seed,

        # ========================================
        # Risk-aware 설정
        # ========================================

        "critical_depth_m":
            0.30,

        "underprediction_weight":
            2.0,
    }

    config["max_epochs"] = args.max_epochs
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

    scaling_source = FoldScalerSource(args.data_dir)

    cv_summary_path = args.cv_dir / "cv_summary.json"
    try:
        cv_summary = json.loads(cv_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"Cannot read CV summary: {exc}")
    cv_critical_depth = cv_summary.get("critical_depth_m")
    if (cv_summary.get("schema_version") != CV_SCHEMA_VERSION
            or cv_summary.get("data_fingerprint") != scaling_source.fingerprint
            or cv_summary.get("evaluation_scope") != "development_only"
            or cv_summary.get("sealed_test_used") is not False
            or cv_summary.get("critical_threshold_role") != "external_depth_crossing"
            or isinstance(cv_critical_depth, bool)
            or not isinstance(cv_critical_depth, (int, float))
            or not np.isfinite(cv_critical_depth)
            or not np.isclose(cv_critical_depth, DEFAULT_CRITICAL_DEPTH_M,
                              rtol=0., atol=1e-12)):
        parser.error("CV split summary does not match the current preprocessing artifacts")
    cv_fold_records = {row.get("fold"): row for row in cv_summary.get("folds", [])
                       if isinstance(row, dict)}
    cv_split_summary_sha256 = sha256(cv_summary_path)

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
        record = cv_fold_records.get(fold_number, {}).get("artifact", {})
        if (record.get("path") != fold_path.name
                or record.get("sha256") != sha256(fold_path)):
            raise ValueError(f"CV fold {fold_number} does not match cv_summary.json")

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

        fold_scaler = scaling_source.fit(train_indices)
        fold_data = rescale_data(data, scaler, fold_scaler)
        provenance = {
            "data_fingerprint": scaling_source.fingerprint,
            "cv_split_summary_sha256": cv_split_summary_sha256,
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
            {
                "schema_version": CV_SCHEMA_VERSION,
                "data_fingerprint": scaling_source.fingerprint,
                "evaluation_scope": "development_cv_validation",
                "sealed_test_used": False,
                "critical_threshold_role": "external_depth_crossing",
                "cv_split_summary_sha256": cv_split_summary_sha256,
                "seed": config["seed"],
                "config": config,
                "folds": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
