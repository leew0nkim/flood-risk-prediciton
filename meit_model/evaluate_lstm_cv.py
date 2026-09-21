from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from meit_model.fold_scaling import checkpoint_data, data_fingerprint

from meit_data.torch_data import (
    depth_to_m,
    horizon_errors,
)

from meit_model.cv import (
    load_development_data,
    load_scaler,
)

from meit_model.lstm import (
    LSTMForecaster,
)


DEFAULT_CRITICAL_DEPTH_M = 0.30


def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return None

    return numerator / denominator


def load_checkpoint_model(
    checkpoint_path: Path,
    device,
):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # 구버전 PyTorch 호환
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    config = checkpoint["config"]

    model = LSTMForecaster(
        input_size=config.get(
            "input_size",
            1,
        ),
        hidden_size=config.get(
            "hidden_size",
            64,
        ),
        num_layers=1,
        horizon=config.get(
            "horizon",
            15,
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model, checkpoint


@torch.no_grad()
def evaluate_fold(
    model,
    data,
    indices,
    scaler,
    device,
    critical_depth_m,
    batch_size=128,
):

    # ========================================================
    # 일반 예측 성능
    # ========================================================

    absolute_error_sum = torch.zeros(
        15,
        dtype=torch.float64,
    )

    squared_error_sum = torch.zeros(
        15,
        dtype=torch.float64,
    )

    observed_count = torch.zeros(
        15,
        dtype=torch.int64,
    )

    # ========================================================
    # High-water 위험 구간
    # actual >= 0.30m
    # ========================================================

    high_water_count = 0

    high_water_absolute_error_sum_m = 0.0

    high_water_bias_sum_m = 0.0

    underprediction_count = 0

    underprediction_gap_sum_m = 0.0

    # ========================================================
    # Point-level threshold classification
    #
    # 미래 각 +1 ~ +15분 시점을 독립적으로 평가
    # ========================================================

    point_tp = 0
    point_fp = 0
    point_fn = 0
    point_tn = 0

    # ========================================================
    # Sample-level threshold crossing
    #
    # 현재 < 0.30m인 상태에서
    # 미래 15분 안에 처음 0.30m를 넘는지 평가
    #
    # crossing timing의 모호성을 피하기 위해
    # Y 15개가 모두 관측된 sample만 사용
    # ========================================================

    crossing_eligible = 0

    crossing_actual = 0
    crossing_predicted = 0

    crossing_tp = 0
    crossing_fp = 0
    crossing_fn = 0
    crossing_tn = 0

    timing_match_count = 0

    timing_absolute_error_sum_min = 0.0
    timing_signed_error_sum_min = 0.0

    late_count = 0
    early_count = 0
    on_time_count = 0

    late_or_missed_count = 0

    # ========================================================
    # Batch evaluation
    # ========================================================

    for start in range(
        0,
        len(indices),
        batch_size,
    ):

        batch_indices = indices[
            start:start + batch_size
        ]

        X = torch.from_numpy(
            np.ascontiguousarray(
                data.X[batch_indices]
            )
        ).float().to(device)

        Y = torch.from_numpy(
            np.ascontiguousarray(
                data.Y[batch_indices]
            )
        ).float().to(device)

        Y_mask = torch.from_numpy(
            np.ascontiguousarray(
                data.Y_mask[batch_indices]
            )
        ).bool().to(device)

        prediction = model(X)

        if prediction.shape != Y.shape:
            raise RuntimeError(
                "Prediction shape mismatch: "
                f"{prediction.shape} "
                f"vs {Y.shape}"
            )

        # ----------------------------------------------------
        # metre 단위 복원
        #
        # handoff 규칙 그대로:
        # prediction -> negative clip
        # target     -> clip하지 않음
        # ----------------------------------------------------

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

        current_m = depth_to_m(
            X[:, -1:, :],
            scaler,
            clip_negative=False,
        )[:, 0, 0]

        # ----------------------------------------------------
        # 일반 MAE / RMSE
        # ----------------------------------------------------

        stats = horizon_errors(
            prediction_m,
            target_m,
            Y_mask,
        )

        absolute_error_sum += (
            stats[
                "absolute_error_sum_m"
            ].cpu()
        )

        squared_error_sum += (
            stats[
                "squared_error_sum_m2"
            ].cpu()
        )

        observed_count += (
            stats[
                "observed_count"
            ].cpu()
        )

        valid = (
            Y_mask
            & torch.isfinite(
                target_m
            )
        )

        # ====================================================
        # High-water evaluation
        # ====================================================

        high_water = (
            valid
            & (
                target_m
                >= critical_depth_m
            )
        )

        if high_water.any():

            actual_high = (
                target_m[
                    high_water
                ]
            )

            predicted_high = (
                prediction_m[
                    high_water
                ]
            )

            difference = (
                predicted_high
                - actual_high
            )

            high_water_count += int(
                high_water.sum().item()
            )

            high_water_absolute_error_sum_m += (
                difference
                .abs()
                .sum()
                .item()
            )

            # prediction - actual
            #
            # 음수이면 전체적으로
            # 과소예측하는 방향
            high_water_bias_sum_m += (
                difference
                .sum()
                .item()
            )

            under = (
                predicted_high
                < actual_high
            )

            underprediction_count += int(
                under.sum().item()
            )

            if under.any():

                underprediction_gap_sum_m += (
                    (
                        actual_high[under]
                        - predicted_high[under]
                    )
                    .sum()
                    .item()
                )

        # ====================================================
        # Point-level 0.30m classification
        # ====================================================

        actual_positive = (
            valid
            & (
                target_m
                >= critical_depth_m
            )
        )

        predicted_positive = (
            valid
            & (
                prediction_m
                >= critical_depth_m
            )
        )

        point_tp += int(
            (
                actual_positive
                & predicted_positive
            ).sum().item()
        )

        point_fn += int(
            (
                actual_positive
                & ~predicted_positive
            ).sum().item()
        )

        point_fp += int(
            (
                ~actual_positive
                & predicted_positive
                & valid
            ).sum().item()
        )

        point_tn += int(
            (
                ~actual_positive
                & ~predicted_positive
                & valid
            ).sum().item()
        )

        # ====================================================
        # Sample-level threshold crossing
        # ====================================================

        valid_sequence = valid[
            :,
            :,
            0,
        ]

        fully_observed = (
            valid_sequence.all(
                dim=1
            )
        )

        actual_sequence = (
            target_m[
                :,
                :,
                0,
            ]
        )

        predicted_sequence = (
            prediction_m[
                :,
                :,
                0,
            ]
        )

        # 현재 이미 위험수위 이상인 sample은
        # "미래 crossing 예측" 평가에서 제외
        eligible = (
            fully_observed
            & (
                current_m
                < critical_depth_m
            )
        )

        actual_cross_any = (
            actual_sequence
            >= critical_depth_m
        ).any(dim=1)

        predicted_cross_any = (
            predicted_sequence
            >= critical_depth_m
        ).any(dim=1)

        actual_cross = (
            eligible
            & actual_cross_any
        )

        predicted_cross = (
            eligible
            & predicted_cross_any
        )

        crossing_eligible += int(
            eligible.sum().item()
        )

        crossing_actual += int(
            actual_cross.sum().item()
        )

        crossing_predicted += int(
            predicted_cross.sum().item()
        )

        crossing_tp += int(
            (
                actual_cross
                & predicted_cross
            ).sum().item()
        )

        crossing_fn += int(
            (
                actual_cross
                & ~predicted_cross
            ).sum().item()
        )

        crossing_fp += int(
            (
                ~actual_cross
                & predicted_cross
                & eligible
            ).sum().item()
        )

        crossing_tn += int(
            (
                ~actual_cross
                & ~predicted_cross
                & eligible
            ).sum().item()
        )

        # ----------------------------------------------------
        # 실제/예측 첫 threshold crossing 시각
        #
        # 0번 index = +1분
        # ----------------------------------------------------

        actual_first = (
            (
                actual_sequence
                >= critical_depth_m
            )
            .to(torch.int64)
            .argmax(dim=1)
            + 1
        )

        predicted_first = (
            (
                predicted_sequence
                >= critical_depth_m
            )
            .to(torch.int64)
            .argmax(dim=1)
            + 1
        )

        both_cross = (
            actual_cross
            & predicted_cross
        )

        if both_cross.any():

            timing_error = (
                predicted_first[
                    both_cross
                ]
                - actual_first[
                    both_cross
                ]
            )

            timing_match_count += int(
                both_cross.sum().item()
            )

            timing_absolute_error_sum_min += (
                timing_error
                .abs()
                .sum()
                .item()
            )

            timing_signed_error_sum_min += (
                timing_error
                .sum()
                .item()
            )

            late_count += int(
                (
                    timing_error > 0
                ).sum().item()
            )

            early_count += int(
                (
                    timing_error < 0
                ).sum().item()
            )

            on_time_count += int(
                (
                    timing_error == 0
                ).sum().item()
            )

        # 실제 crossing인데
        #
        # 1. 예측이 아예 crossing하지 못했거나
        # 2. 실제보다 늦게 crossing을 예측
        #
        # 한 경우를 safety 관점의
        # late-or-missed로 계산
        late_or_missed = (
            actual_cross
            & (
                ~predicted_cross_any
                |
                (
                    predicted_first
                    > actual_first
                )
            )
        )

        late_or_missed_count += int(
            late_or_missed.sum().item()
        )

    # ========================================================
    # 일반 metrics
    # ========================================================

    if (
        observed_count == 0
    ).any():
        raise RuntimeError(
            "A forecast horizon "
            "has no observed targets"
        )

    horizon_mae_m = (
        absolute_error_sum
        / observed_count
    )

    horizon_rmse_m = torch.sqrt(
        squared_error_sum
        / observed_count
    )

    total_count = int(
        observed_count
        .sum()
        .item()
    )

    overall_mae_m = float(
        absolute_error_sum
        .sum()
        / total_count
    )

    overall_rmse_m = float(
        torch.sqrt(
            squared_error_sum
            .sum()
            / total_count
        )
    )

    # ========================================================
    # Risk metrics
    # ========================================================

    point_precision = safe_divide(
        point_tp,
        point_tp + point_fp,
    )

    point_recall = safe_divide(
        point_tp,
        point_tp + point_fn,
    )

    crossing_precision = safe_divide(
        crossing_tp,
        crossing_tp + crossing_fp,
    )

    crossing_recall = safe_divide(
        crossing_tp,
        crossing_tp + crossing_fn,
    )

    if (
        crossing_precision is not None
        and crossing_recall is not None
        and (
            crossing_precision
            + crossing_recall
        ) > 0
    ):
        crossing_f1 = (
            2
            * crossing_precision
            * crossing_recall
            / (
                crossing_precision
                + crossing_recall
            )
        )
    else:
        crossing_f1 = None

    return {
        "overall_mae_cm":
            overall_mae_m * 100,

        "overall_rmse_cm":
            overall_rmse_m * 100,

        "horizon_mae_cm":
            (
                horizon_mae_m
                * 100
            ).tolist(),

        "horizon_rmse_cm":
            (
                horizon_rmse_m
                * 100
            ).tolist(),

        "observed_count":
            observed_count.tolist(),

        "risk": {
            "critical_depth_m":
                critical_depth_m,

            "high_water_points":
                high_water_count,

            "high_water_mae_cm":
                (
                    safe_divide(
                        high_water_absolute_error_sum_m,
                        high_water_count,
                    )
                    * 100
                    if high_water_count
                    else None
                ),

            "high_water_bias_cm":
                (
                    safe_divide(
                        high_water_bias_sum_m,
                        high_water_count,
                    )
                    * 100
                    if high_water_count
                    else None
                ),

            "underprediction_count":
                underprediction_count,

            "underprediction_rate":
                safe_divide(
                    underprediction_count,
                    high_water_count,
                ),

            "mean_underprediction_cm":
                (
                    safe_divide(
                        underprediction_gap_sum_m,
                        underprediction_count,
                    )
                    * 100
                    if underprediction_count
                    else None
                ),

            "point_threshold_tp":
                point_tp,

            "point_threshold_fp":
                point_fp,

            "point_threshold_fn":
                point_fn,

            "point_threshold_tn":
                point_tn,

            "point_threshold_precision":
                point_precision,

            "point_threshold_recall":
                point_recall,

            "crossing_eligible_samples":
                crossing_eligible,

            "actual_crossing_samples":
                crossing_actual,

            "predicted_crossing_samples":
                crossing_predicted,

            "crossing_tp":
                crossing_tp,

            "crossing_fp":
                crossing_fp,

            "crossing_fn":
                crossing_fn,

            "crossing_tn":
                crossing_tn,

            "crossing_precision":
                crossing_precision,

            "crossing_recall":
                crossing_recall,

            "crossing_f1":
                crossing_f1,

            "crossing_timing_matches":
                timing_match_count,

            "crossing_timing_mae_min":
                safe_divide(
                    timing_absolute_error_sum_min,
                    timing_match_count,
                ),

            # 양수 = 실제보다 늦게 예측
            # 음수 = 실제보다 일찍 예측
            "crossing_timing_bias_min":
                safe_divide(
                    timing_signed_error_sum_min,
                    timing_match_count,
                ),

            "early_crossing_predictions":
                early_count,

            "on_time_crossing_predictions":
                on_time_count,

            "late_crossing_predictions":
                late_count,

            "late_or_missed_count":
                late_or_missed_count,

            "late_or_missed_rate":
                safe_divide(
                    late_or_missed_count,
                    crossing_actual,
                ),
        },
    }


def format_optional(
    value,
    digits=4,
):
    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


def print_fold_result(
    fold_number,
    checkpoint,
    result,
):

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

    print(
        "Best epoch:",
        checkpoint["epoch"],
    )

    print(
        f"Overall MAE : "
        f"{result['overall_mae_cm']:.4f} cm"
    )

    print(
        f"Overall RMSE: "
        f"{result['overall_rmse_cm']:.4f} cm"
    )

    for horizon in (
        5,
        10,
        15,
    ):

        i = horizon - 1

        print(
            f"+{horizon:2d} min "
            f"MAE="
            f"{result['horizon_mae_cm'][i]:.4f} cm, "
            f"RMSE="
            f"{result['horizon_rmse_cm'][i]:.4f} cm"
        )

    risk = result["risk"]

    print(
        "\n"
        "--- Risk-aware ---"
    )

    print(
        "High-water points:",
        risk[
            "high_water_points"
        ],
    )

    print(
        "High-water MAE:",
        format_optional(
            risk[
                "high_water_mae_cm"
            ]
        ),
        "cm",
    )

    print(
        "High-water bias:",
        format_optional(
            risk[
                "high_water_bias_cm"
            ]
        ),
        "cm",
    )

    print(
        "Underprediction rate:",
        format_optional(
            (
                risk[
                    "underprediction_rate"
                ]
                * 100
            )
            if risk[
                "underprediction_rate"
            ] is not None
            else None,
            2,
        ),
        "%",
    )

    print(
        "Mean underprediction:",
        format_optional(
            risk[
                "mean_underprediction_cm"
            ]
        ),
        "cm",
    )

    print(
        "Point threshold recall:",
        format_optional(
            (
                risk[
                    "point_threshold_recall"
                ]
                * 100
            )
            if risk[
                "point_threshold_recall"
            ] is not None
            else None,
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

    print(
        "Crossing recall:",
        format_optional(
            (
                risk[
                    "crossing_recall"
                ]
                * 100
            )
            if risk[
                "crossing_recall"
            ] is not None
            else None,
            2,
        ),
        "%",
    )

    print(
        "Crossing precision:",
        format_optional(
            (
                risk[
                    "crossing_precision"
                ]
                * 100
            )
            if risk[
                "crossing_precision"
            ] is not None
            else None,
            2,
        ),
        "%",
    )

    print(
        "Crossing timing MAE:",
        format_optional(
            risk[
                "crossing_timing_mae_min"
            ],
            2,
        ),
        "min",
    )

    print(
        "Crossing timing bias:",
        format_optional(
            risk[
                "crossing_timing_bias_min"
            ],
            2,
        ),
        "min",
    )

    print(
        "Late or missed rate:",
        format_optional(
            (
                risk[
                    "late_or_missed_rate"
                ]
                * 100
            )
            if risk[
                "late_or_missed_rate"
            ] is not None
            else None,
            2,
        ),
        "%",
    )


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
        "--model-dir",
        type=Path,
        default=Path(
            "artifacts/lstm_cv_fold_scaled"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lstm_cv_fold_scaled/"
            "evaluation.json"
        ),
    )

    parser.add_argument(
        "--critical-depth",
        type=float,
        default=(
            DEFAULT_CRITICAL_DEPTH_M
        ),
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

    fingerprint = data_fingerprint(args.data_dir)
    scaler = load_scaler(
        args.data_dir
    )

    results = []

    for fold_number in range(
        1,
        5,
    ):

        fold_path = (
            args.cv_dir
            / f"fold_{fold_number}.npz"
        )

        checkpoint_path = (
            args.model_dir
            / f"fold_{fold_number}"
            / "best_model.pt"
        )

        if not fold_path.exists():
            raise FileNotFoundError(
                fold_path
            )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                checkpoint_path
            )

        with np.load(
            fold_path,
            allow_pickle=False,
        ) as fold:

            val_indices = (
                fold["val_indices"]
            )

        model, checkpoint = (
            load_checkpoint_model(
                checkpoint_path,
                device,
            )
        )

        fold_data, fold_scaler = checkpoint_data(
            data, scaler, checkpoint, fingerprint, val_indices
        )

        result = evaluate_fold(
            model=model,
            data=fold_data,
            indices=val_indices,
            scaler=fold_scaler,
            device=device,
            critical_depth_m=(
                args.critical_depth
            ),
            batch_size=args.batch_size,
        )

        print_fold_result(
            fold_number,
            checkpoint,
            result,
        )

        results.append(
            {
                "fold":
                    fold_number,

                "best_epoch":
                    int(
                        checkpoint[
                            "epoch"
                        ]
                    ),

                **result,
            }
        )

    # ========================================================
    # CV summary
    # ========================================================

    mae = np.array(
        [
            result[
                "overall_mae_cm"
            ]
            for result in results
        ]
    )

    rmse = np.array(
        [
            result[
                "overall_rmse_cm"
            ]
            for result in results
        ]
    )

    crossing_tp = sum(
        r["risk"]["crossing_tp"]
        for r in results
    )

    crossing_fn = sum(
        r["risk"]["crossing_fn"]
        for r in results
    )

    crossing_fp = sum(
        r["risk"]["crossing_fp"]
        for r in results
    )

    crossing_actual = (
        crossing_tp
        + crossing_fn
    )

    pooled_crossing_recall = (
        safe_divide(
            crossing_tp,
            crossing_actual,
        )
    )

    pooled_crossing_precision = (
        safe_divide(
            crossing_tp,
            crossing_tp
            + crossing_fp,
        )
    )

    late_or_missed = sum(
        r["risk"][
            "late_or_missed_count"
        ]
        for r in results
    )

    pooled_late_or_missed_rate = (
        safe_divide(
            late_or_missed,
            crossing_actual,
        )
    )

    summary = {
        "mae_mean_cm":
            float(
                mae.mean()
            ),

        "mae_std_cm":
            float(
                mae.std(
                    ddof=1
                )
            ),

        "rmse_mean_cm":
            float(
                rmse.mean()
            ),

        "rmse_std_cm":
            float(
                rmse.std(
                    ddof=1
                )
            ),

        "pooled_crossing_tp":
            crossing_tp,

        "pooled_crossing_fp":
            crossing_fp,

        "pooled_crossing_fn":
            crossing_fn,

        "pooled_crossing_recall":
            pooled_crossing_recall,

        "pooled_crossing_precision":
            pooled_crossing_precision,

        "pooled_late_or_missed_rate":
            pooled_late_or_missed_rate,
    }

    print(
        "\n"
        "================================"
    )

    print(
        "LSTM CV SUMMARY"
    )

    print(
        "================================"
    )

    print(
        f"MAE : "
        f"{summary['mae_mean_cm']:.4f} "
        f"± {summary['mae_std_cm']:.4f} cm"
    )

    print(
        f"RMSE: "
        f"{summary['rmse_mean_cm']:.4f} "
        f"± {summary['rmse_std_cm']:.4f} cm"
    )

    print(
        "Pooled crossing recall:",
        format_optional(
            (
                pooled_crossing_recall
                * 100
            )
            if pooled_crossing_recall
            is not None
            else None,
            2,
        ),
        "%",
    )

    print(
        "Pooled crossing precision:",
        format_optional(
            (
                pooled_crossing_precision
                * 100
            )
            if pooled_crossing_precision
            is not None
            else None,
            2,
        ),
        "%",
    )

    print(
        "Pooled late-or-missed rate:",
        format_optional(
            (
                pooled_late_or_missed_rate
                * 100
            )
            if pooled_late_or_missed_rate
            is not None
            else None,
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

                "folds":
                    results,

                "summary":
                    summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nDetailed LSTM evaluation passed."
    )


if __name__ == "__main__":
    main()