from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from meit_model.fold_scaling import checkpoint_data, data_fingerprint

from meit_data.torch_data import depth_to_m

from meit_model.baselines import linear_forecast
from meit_model.cv import (
    load_development_data,
    load_scaler,
)
from meit_model.evaluate_lstm_cv import (
    load_checkpoint_model,
)


ACTUAL_CRITICAL_DEPTH_M = 0.30

DEFAULT_ALARM_THRESHOLDS = [
    0.300,
    0.295,
    0.290,
    0.285,
    0.280,
]


def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return None

    return numerator / denominator


@torch.no_grad()
def collect_predictions(
    model,
    data,
    indices,
    scaler,
    device,
    batch_size=128,
):
    """
    한 fold의 LSTM prediction / Linear prediction /
    실제 target을 모두 metre 단위로 수집한다.
    """

    current_list = []
    actual_list = []
    mask_list = []

    lstm_list = []
    linear_list = []

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

        # --------------------------------------------
        # LSTM
        # --------------------------------------------

        lstm_prediction = model(X)

        # --------------------------------------------
        # 최근 5분 Linear
        # --------------------------------------------

        linear_prediction = linear_forecast(
            X,
            history=5,
            horizon=15,
        )

        # --------------------------------------------
        # metre 단위 복원
        # --------------------------------------------

        current_m = depth_to_m(
            X[:, -1:, :],
            scaler,
            clip_negative=False,
        )[:, 0, 0]

        actual_m = depth_to_m(
            Y,
            scaler,
            clip_negative=False,
        )[:, :, 0]

        lstm_m = depth_to_m(
            lstm_prediction,
            scaler,
            clip_negative=True,
        )[:, :, 0]

        linear_m = depth_to_m(
            linear_prediction,
            scaler,
            clip_negative=True,
        )[:, :, 0]

        current_list.append(
            current_m.cpu()
        )

        actual_list.append(
            actual_m.cpu()
        )

        mask_list.append(
            Y_mask[:, :, 0].cpu()
        )

        lstm_list.append(
            lstm_m.cpu()
        )

        linear_list.append(
            linear_m.cpu()
        )

    return {
        "current_m":
            torch.cat(
                current_list,
                dim=0,
            ),

        "actual_m":
            torch.cat(
                actual_list,
                dim=0,
            ),

        "mask":
            torch.cat(
                mask_list,
                dim=0,
            ),

        "lstm_m":
            torch.cat(
                lstm_list,
                dim=0,
            ),

        "linear_m":
            torch.cat(
                linear_list,
                dim=0,
            ),
    }


def evaluate_alarm(
    current_m,
    actual_m,
    mask,
    predicted_alarm_sequence,
    alarm_threshold_m,
    actual_critical_m,
):
    """
    실제 위험 기준은 actual_critical_m.
    예측 경고 기준만 alarm_threshold_m을 사용한다.
    """

    valid = (
        mask
        & torch.isfinite(
            actual_m
        )
    )

    # crossing 시각 계산이 가능하도록
    # 미래 15개 target이 모두 관측된 경우만 사용
    fully_observed = valid.all(
        dim=1
    )

    # 이미 현재 수위가 0.30m 이상이면
    # "미래 crossing 사전경고" 평가 대상이 아님
    eligible = (
        fully_observed
        & (
            current_m
            < actual_critical_m
        )
    )

    # --------------------------------------------
    # 실제 0.30m crossing
    # --------------------------------------------

    actual_positive_sequence = (
        actual_m
        >= actual_critical_m
    )

    actual_cross_any = (
        actual_positive_sequence.any(
            dim=1
        )
    )

    actual_cross = (
        eligible
        & actual_cross_any
    )

    # --------------------------------------------
    # 예측 alarm threshold crossing
    # --------------------------------------------

    predicted_alarm_sequence_bool = (
        predicted_alarm_sequence
        >= alarm_threshold_m
    )

    predicted_alarm_any = (
        predicted_alarm_sequence_bool.any(
            dim=1
        )
    )

    predicted_alarm = (
        eligible
        & predicted_alarm_any
    )

    # --------------------------------------------
    # Sample-level TP / FP / FN / TN
    # --------------------------------------------

    tp = int(
        (
            actual_cross
            & predicted_alarm
        ).sum().item()
    )

    fp = int(
        (
            ~actual_cross
            & predicted_alarm
            & eligible
        ).sum().item()
    )

    fn = int(
        (
            actual_cross
            & ~predicted_alarm
        ).sum().item()
    )

    tn = int(
        (
            ~actual_cross
            & ~predicted_alarm
            & eligible
        ).sum().item()
    )

    # --------------------------------------------
    # 첫 crossing / alarm 시각
    #
    # index 0 = +1분
    # --------------------------------------------

    actual_first = (
        actual_positive_sequence
        .to(torch.int64)
        .argmax(dim=1)
        + 1
    )

    alarm_first = (
        predicted_alarm_sequence_bool
        .to(torch.int64)
        .argmax(dim=1)
        + 1
    )

    both = (
        actual_cross
        & predicted_alarm
    )

    # --------------------------------------------
    # lead time
    #
    # actual time - alarm time
    #
    # +3 → 3분 일찍 경고
    #  0 → 정확히 같은 시점
    # -2 → 2분 늦게 경고
    # --------------------------------------------

    lead_time_sum = 0.0
    lead_time_abs_sum = 0.0
    timing_count = 0

    early_count = 0
    on_time_count = 0
    late_count = 0

    if both.any():

        lead_time = (
            actual_first[both]
            - alarm_first[both]
        )

        timing_count = int(
            both.sum().item()
        )

        lead_time_sum = float(
            lead_time.sum().item()
        )

        lead_time_abs_sum = float(
            lead_time.abs().sum().item()
        )

        early_count = int(
            (
                lead_time > 0
            ).sum().item()
        )

        on_time_count = int(
            (
                lead_time == 0
            ).sum().item()
        )

        late_count = int(
            (
                lead_time < 0
            ).sum().item()
        )

    # --------------------------------------------
    # 제시간 경고
    #
    # 실제 crossing이 발생했고,
    # alarm이 실제 crossing 이전 또는 같은 시점
    # --------------------------------------------

    timely = (
        actual_cross
        & predicted_alarm
        & (
            alarm_first
            <= actual_first
        )
    )

    timely_count = int(
        timely.sum().item()
    )

    actual_count = int(
        actual_cross.sum().item()
    )

    late_or_missed_count = (
        actual_count
        - timely_count
    )

    return {
        "eligible_samples":
            int(
                eligible.sum().item()
            ),

        "actual_crossings":
            actual_count,

        "predicted_alarms":
            int(
                predicted_alarm.sum().item()
            ),

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "tn":
            tn,

        "recall":
            safe_divide(
                tp,
                tp + fn,
            ),

        "precision":
            safe_divide(
                tp,
                tp + fp,
            ),

        "timely_count":
            timely_count,

        "timely_recall":
            safe_divide(
                timely_count,
                actual_count,
            ),

        "late_or_missed_count":
            late_or_missed_count,

        "late_or_missed_rate":
            safe_divide(
                late_or_missed_count,
                actual_count,
            ),

        "mean_lead_time_min":
            safe_divide(
                lead_time_sum,
                timing_count,
            ),

        "timing_mae_min":
            safe_divide(
                lead_time_abs_sum,
                timing_count,
            ),

        "early_count":
            early_count,

        "on_time_count":
            on_time_count,

        "late_count":
            late_count,
    }


def pool_results(
    fold_results,
):
    keys_to_sum = [
        "eligible_samples",
        "actual_crossings",
        "predicted_alarms",
        "tp",
        "fp",
        "fn",
        "tn",
        "timely_count",
        "late_or_missed_count",
        "early_count",
        "on_time_count",
        "late_count",
    ]

    pooled = {
        key: sum(
            result[key]
            for result in fold_results
        )
        for key in keys_to_sum
    }

    pooled["recall"] = safe_divide(
        pooled["tp"],
        pooled["tp"]
        + pooled["fn"],
    )

    pooled["precision"] = safe_divide(
        pooled["tp"],
        pooled["tp"]
        + pooled["fp"],
    )

    pooled["timely_recall"] = (
        safe_divide(
            pooled["timely_count"],
            pooled["actual_crossings"],
        )
    )

    pooled["late_or_missed_rate"] = (
        safe_divide(
            pooled[
                "late_or_missed_count"
            ],
            pooled[
                "actual_crossings"
            ],
        )
    )

    pooled["false_positive_rate"] = safe_divide(
        pooled["fp"],
        pooled["fp"] + pooled["tn"],
    )

    pooled["alarm_rate"] = safe_divide(
        pooled["predicted_alarms"],
        pooled["eligible_samples"],
    )

    return pooled


def percentage(
    value,
):
    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


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
            "artifacts/risk_calibration_fold_scaled/"
            "warning_threshold_sweep.json"
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
        "Actual critical depth:",
        ACTUAL_CRITICAL_DEPTH_M,
        "m",
    )

    data = load_development_data(
        args.data_dir
    )

    fingerprint = data_fingerprint(args.data_dir)
    scaler = load_scaler(
        args.data_dir
    )

    # ========================================================
    # 먼저 fold별 prediction을 한 번만 계산
    # threshold마다 inference를 다시 하지 않는다.
    # ========================================================

    fold_predictions = {}

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

        with np.load(
            fold_path,
            allow_pickle=False,
        ) as fold:

            val_indices = (
                fold["val_indices"]
            )

        model, checkpoint = load_checkpoint_model(
            checkpoint_path,
            device,
        )

        fold_data, fold_scaler = checkpoint_data(
            data, scaler, checkpoint, fingerprint, val_indices
        )

        fold_predictions[
            fold_number
        ] = collect_predictions(
            model=model,
            data=fold_data,
            indices=val_indices,
            scaler=fold_scaler,
            device=device,
            batch_size=args.batch_size,
        )

    # ========================================================
    # 1. LSTM alarm threshold sweep
    # ========================================================

    sweep_results = []

    print(
        "\n"
        "================================"
    )
    print(
        "LSTM WARNING THRESHOLD SWEEP"
    )
    print(
        "================================"
    )

    for threshold in (
        DEFAULT_ALARM_THRESHOLDS
    ):

        fold_results = []

        for fold_number in range(
            1,
            5,
        ):

            p = fold_predictions[
                fold_number
            ]

            result = evaluate_alarm(
                current_m=p[
                    "current_m"
                ],
                actual_m=p[
                    "actual_m"
                ],
                mask=p["mask"],
                predicted_alarm_sequence=p[
                    "lstm_m"
                ],
                alarm_threshold_m=threshold,
                actual_critical_m=(
                    ACTUAL_CRITICAL_DEPTH_M
                ),
            )

            fold_results.append(
                result
            )

        pooled = pool_results(
            fold_results
        )

        sweep_results.append(
            {
                "alarm_threshold_m":
                    threshold,

                "folds":
                    fold_results,

                "pooled":
                    pooled,
            }
        )

        print(
            f"\nAlarm threshold: "
            f"{threshold:.3f} m"
        )

        print(
            "  Recall         :",
            percentage(
                pooled["recall"]
            ),
        )

        print(
            "  Precision      :",
            percentage(
                pooled["precision"]
            ),
        )

        print(
            "  Timely recall  :",
            percentage(
                pooled[
                    "timely_recall"
                ]
            ),
        )

        print(
            "  Late or missed :",
            percentage(
                pooled[
                    "late_or_missed_rate"
                ]
            ),
        )

        print(
            "  False alarms   :",
            pooled["fp"],
        )

        print(
            "  FPR            :",
            percentage(pooled["false_positive_rate"]),
        )

        print(
            "  Alarm rate     :",
            percentage(pooled["alarm_rate"]),
        )

    # ========================================================
    # 2. LSTM + Linear safety guard
    #
    # LSTM은 각 calibration threshold,
    # Linear는 실제 critical 0.30m 기준 유지.
    #
    # 둘 중 하나라도 경고하면 alarm.
    # ========================================================

    hybrid_results = []

    print(
        "\n"
        "================================"
    )
    print(
        "LSTM + LINEAR SAFETY GUARD"
    )
    print(
        "================================"
    )

    for threshold in (
        DEFAULT_ALARM_THRESHOLDS
    ):

        fold_results = []

        for fold_number in range(
            1,
            5,
        ):

            p = fold_predictions[
                fold_number
            ]

            # ----------------------------------------
            # LSTM alarm
            # OR
            # Linear >= 0.30m
            #
            # bool OR을 한 뒤 evaluate_alarm에서
            # threshold 비교를 다시 하는 대신,
            # True -> 1.0
            # False -> 0.0
            #
            # threshold=0.5로 평가
            # ----------------------------------------

            hybrid_alarm = (
                (
                    p["lstm_m"]
                    >= threshold
                )
                |
                (
                    p["linear_m"]
                    >= ACTUAL_CRITICAL_DEPTH_M
                )
            ).float()

            result = evaluate_alarm(
                current_m=p[
                    "current_m"
                ],
                actual_m=p[
                    "actual_m"
                ],
                mask=p["mask"],
                predicted_alarm_sequence=(
                    hybrid_alarm
                ),
                alarm_threshold_m=0.5,
                actual_critical_m=(
                    ACTUAL_CRITICAL_DEPTH_M
                ),
            )

            fold_results.append(
                result
            )

        pooled = pool_results(
            fold_results
        )

        hybrid_results.append(
            {
                "lstm_alarm_threshold_m":
                    threshold,

                "linear_alarm_threshold_m":
                    ACTUAL_CRITICAL_DEPTH_M,

                "folds":
                    fold_results,

                "pooled":
                    pooled,
            }
        )

        print(
            f"\nLSTM {threshold:.3f} "
            f"OR Linear 0.300"
        )

        print(
            "  Recall         :",
            percentage(
                pooled["recall"]
            ),
        )

        print(
            "  Precision      :",
            percentage(
                pooled["precision"]
            ),
        )

        print(
            "  Timely recall  :",
            percentage(
                pooled[
                    "timely_recall"
                ]
            ),
        )

        print(
            "  Late or missed :",
            percentage(
                pooled[
                    "late_or_missed_rate"
                ]
            ),
        )

        print(
            "  False alarms   :",
            pooled["fp"],
        )

        print(
            "  FPR            :",
            percentage(pooled["false_positive_rate"]),
        )

        print(
            "  Alarm rate     :",
            percentage(pooled["alarm_rate"]),
        )

    # ========================================================
    # 저장
    # ========================================================

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
                "actual_critical_depth_m":
                    ACTUAL_CRITICAL_DEPTH_M,

                "lstm_threshold_sweep":
                    sweep_results,

                "hybrid_safety_guard":
                    hybrid_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nRisk calibration passed."
    )


if __name__ == "__main__":
    main()
