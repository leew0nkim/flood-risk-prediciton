from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from meit_model.baselines import (
    persistence_forecast,
    linear_forecast,
)

from meit_model.cv import (
    load_development_data,
    load_scaler,
)

from meit_model.evaluate_lstm_cv import (
    evaluate_fold,
    format_optional,
)


DEFAULT_CRITICAL_DEPTH_M = 0.30


class BaselineModel(nn.Module):
    """
    기존 baseline 함수를 nn.Module처럼 감싸서
    LSTM과 완전히 동일한 evaluate_fold()를 사용한다.
    """

    def __init__(self, forecast_fn):
        super().__init__()
        self.forecast_fn = forecast_fn

    def forward(self, X):
        return self.forecast_fn(X)


def print_result(
    model_name,
    fold_number,
    result,
):

    risk = result["risk"]

    print(
        "\n"
        "================================"
    )
    print(
        f"{model_name} - FOLD {fold_number}"
    )
    print(
        "================================"
    )

    print(
        f"Overall MAE : "
        f"{result['overall_mae_cm']:.4f} cm"
    )

    print(
        f"Overall RMSE: "
        f"{result['overall_rmse_cm']:.4f} cm"
    )

    for horizon in (5, 10, 15):

        i = horizon - 1

        print(
            f"+{horizon:2d} min "
            f"MAE="
            f"{result['horizon_mae_cm'][i]:.4f} cm, "
            f"RMSE="
            f"{result['horizon_rmse_cm'][i]:.4f} cm"
        )

    print(
        "\n"
        "--- Risk-aware ---"
    )

    print(
        "High-water points:",
        risk["high_water_points"],
    )

    print(
        "High-water MAE:",
        format_optional(
            risk["high_water_mae_cm"]
        ),
        "cm",
    )

    print(
        "High-water bias:",
        format_optional(
            risk["high_water_bias_cm"]
        ),
        "cm",
    )

    under_rate = (
        risk["underprediction_rate"]
    )

    print(
        "Underprediction rate:",
        format_optional(
            (
                under_rate * 100
                if under_rate is not None
                else None
            ),
            2,
        ),
        "%",
    )

    print(
        "Mean underprediction:",
        format_optional(
            risk["mean_underprediction_cm"]
        ),
        "cm",
    )

    point_recall = (
        risk["point_threshold_recall"]
    )

    print(
        "Point threshold recall:",
        format_optional(
            (
                point_recall * 100
                if point_recall is not None
                else None
            ),
            2,
        ),
        "%",
    )

    print(
        "Crossing:",
        f"TP={risk['crossing_tp']}, "
        f"FP={risk['crossing_fp']}, "
        f"FN={risk['crossing_fn']}"
    )

    crossing_recall = (
        risk["crossing_recall"]
    )

    crossing_precision = (
        risk["crossing_precision"]
    )

    print(
        "Crossing recall:",
        format_optional(
            (
                crossing_recall * 100
                if crossing_recall is not None
                else None
            ),
            2,
        ),
        "%",
    )

    print(
        "Crossing precision:",
        format_optional(
            (
                crossing_precision * 100
                if crossing_precision is not None
                else None
            ),
            2,
        ),
        "%",
    )

    print(
        "Crossing timing MAE:",
        format_optional(
            risk["crossing_timing_mae_min"],
            2,
        ),
        "min",
    )

    print(
        "Crossing timing bias:",
        format_optional(
            risk["crossing_timing_bias_min"],
            2,
        ),
        "min",
    )

    late_rate = (
        risk["late_or_missed_rate"]
    )

    print(
        "Late or missed rate:",
        format_optional(
            (
                late_rate * 100
                if late_rate is not None
                else None
            ),
            2,
        ),
        "%",
    )


def summarize_model(
    model_name,
    folds,
):

    mae = np.array(
        [
            x["overall_mae_cm"]
            for x in folds
        ]
    )

    rmse = np.array(
        [
            x["overall_rmse_cm"]
            for x in folds
        ]
    )

    tp = sum(
        x["risk"]["crossing_tp"]
        for x in folds
    )

    fp = sum(
        x["risk"]["crossing_fp"]
        for x in folds
    )

    fn = sum(
        x["risk"]["crossing_fn"]
        for x in folds
    )

    late_or_missed = sum(
        x["risk"]["late_or_missed_count"]
        for x in folds
    )

    actual_crossings = (
        tp + fn
    )

    crossing_recall = (
        tp / actual_crossings
        if actual_crossings > 0
        else None
    )

    crossing_precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else None
    )

    late_or_missed_rate = (
        late_or_missed
        / actual_crossings
        if actual_crossings > 0
        else None
    )

    return {
        "model":
            model_name,

        "mae_mean_cm":
            float(mae.mean()),

        "mae_std_cm":
            float(
                mae.std(ddof=1)
            ),

        "rmse_mean_cm":
            float(rmse.mean()),

        "rmse_std_cm":
            float(
                rmse.std(ddof=1)
            ),

        "pooled_crossing_tp":
            tp,

        "pooled_crossing_fp":
            fp,

        "pooled_crossing_fn":
            fn,

        "pooled_crossing_recall":
            crossing_recall,

        "pooled_crossing_precision":
            crossing_precision,

        "pooled_late_or_missed_rate":
            late_or_missed_rate,
    }


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
        "--output",
        type=Path,
        default=Path(
            "artifacts/baselines/"
            "risk_evaluation.json"
        ),
    )

    parser.add_argument(
        "--critical-depth",
        type=float,
        default=DEFAULT_CRITICAL_DEPTH_M,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    args = parser.parse_args()

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
        args.critical_depth,
        "m",
    )

    data = load_development_data(
        args.data_dir
    )

    scaler = load_scaler(
        args.data_dir
    )

    models = {
        "Persistence":
            BaselineModel(
                persistence_forecast
            ).to(device),

        "Linear-5min":
            BaselineModel(
                lambda X:
                linear_forecast(
                    X,
                    history=5,
                )
            ).to(device),
    }

    all_results = {}

    for model_name, model in models.items():

        fold_results = []

        for fold_number in range(
            1,
            5,
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

                val_indices = (
                    fold["val_indices"]
                )

            result = evaluate_fold(
                model=model,
                data=data,
                indices=val_indices,
                scaler=scaler,
                device=device,
                critical_depth_m=(
                    args.critical_depth
                ),
                batch_size=args.batch_size,
            )

            print_result(
                model_name,
                fold_number,
                result,
            )

            fold_results.append(
                result
            )

        all_results[
            model_name
        ] = {
            "folds":
                fold_results,

            "summary":
                summarize_model(
                    model_name,
                    fold_results,
                ),
        }

    print(
        "\n"
        "================================"
    )

    print(
        "BASELINE RISK SUMMARY"
    )

    print(
        "================================"
    )

    for model_name in (
        "Persistence",
        "Linear-5min",
    ):

        summary = (
            all_results[
                model_name
            ]["summary"]
        )

        print(
            f"\n{model_name}"
        )

        print(
            f"MAE : "
            f"{summary['mae_mean_cm']:.4f} "
            f"± "
            f"{summary['mae_std_cm']:.4f} cm"
        )

        print(
            f"RMSE: "
            f"{summary['rmse_mean_cm']:.4f} "
            f"± "
            f"{summary['rmse_std_cm']:.4f} cm"
        )

        recall = (
            summary[
                "pooled_crossing_recall"
            ]
        )

        precision = (
            summary[
                "pooled_crossing_precision"
            ]
        )

        late_rate = (
            summary[
                "pooled_late_or_missed_rate"
            ]
        )

        print(
            "Pooled crossing recall:",
            format_optional(
                (
                    recall * 100
                    if recall is not None
                    else None
                ),
                2,
            ),
            "%",
        )

        print(
            "Pooled crossing precision:",
            format_optional(
                (
                    precision * 100
                    if precision is not None
                    else None
                ),
                2,
            ),
            "%",
        )

        print(
            "Pooled late-or-missed rate:",
            format_optional(
                (
                    late_rate * 100
                    if late_rate is not None
                    else None
                ),
                2,
            ),
            "%",
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "critical_depth_m":
                    args.critical_depth,

                "models":
                    all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nBaseline risk evaluation passed."
    )


if __name__ == "__main__":
    main()
