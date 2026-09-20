from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from meit_data.torch_data import (
    depth_to_m,
    horizon_errors,
)

from meit_model.baselines import (
    persistence_forecast,
    linear_forecast,
)

from meit_model.cv import (
    load_development_data,
    load_scaler,
)


# ============================================================
# 하나의 baseline을 하나의 fold에서 평가
# ============================================================

def evaluate_model(
    model_fn,
    data,
    indices: np.ndarray,
    scaler,
    batch_size: int = 128,
):
    """
    validation indices에 대해 baseline을 평가한다.

    반환:
        overall MAE/RMSE
        horizon별 MAE/RMSE
        관측 정답 개수
    """

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

    # --------------------------------------------------------
    # validation sample을 batch 단위로 처리
    # --------------------------------------------------------

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
        ).float()

        Y = torch.from_numpy(
            np.ascontiguousarray(
                data.Y[batch_indices]
            )
        ).float()

        Y_mask = torch.from_numpy(
            np.ascontiguousarray(
                data.Y_mask[batch_indices]
            )
        ).bool()

        # ----------------------------------------------------
        # standardized space에서 예측
        # ----------------------------------------------------

        with torch.no_grad():
            prediction = model_fn(X)

        if prediction.shape != Y.shape:
            raise RuntimeError(
                "Prediction shape mismatch: "
                f"{prediction.shape} vs {Y.shape}"
            )

        # ----------------------------------------------------
        # m 단위로 역정규화
        #
        # 팀원의 torch_data.py 규칙:
        #
        # prediction:
        #   clip_negative=True
        #
        # target:
        #   clip_negative=False
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

        # ----------------------------------------------------
        # horizon별 additive statistics
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

    # --------------------------------------------------------
    # Horizon별 MAE / RMSE
    # --------------------------------------------------------

    if (observed_count == 0).any():
        raise RuntimeError(
            "A forecast horizon has no observed targets"
        )

    horizon_mae_m = (
        absolute_error_sum
        / observed_count
    )

    horizon_rmse_m = torch.sqrt(
        squared_error_sum
        / observed_count
    )

    # --------------------------------------------------------
    # 전체 MAE / RMSE
    #
    # horizon 평균이 아니라
    # 전체 관측 정답 기준으로 계산한다.
    # --------------------------------------------------------

    total_count = observed_count.sum()

    overall_mae_m = (
        absolute_error_sum.sum()
        / total_count
    )

    overall_rmse_m = torch.sqrt(
        squared_error_sum.sum()
        / total_count
    )

    return {
        "overall_mae_m":
            float(overall_mae_m),

        "overall_rmse_m":
            float(overall_rmse_m),

        "overall_mae_cm":
            float(overall_mae_m * 100),

        "overall_rmse_cm":
            float(overall_rmse_m * 100),

        "horizon_mae_m":
            horizon_mae_m.tolist(),

        "horizon_rmse_m":
            horizon_rmse_m.tolist(),

        "horizon_mae_cm":
            (
                horizon_mae_m * 100
            ).tolist(),

        "horizon_rmse_cm":
            (
                horizon_rmse_m * 100
            ).tolist(),

        "observed_count":
            observed_count.tolist(),
    }


# ============================================================
# 보기 좋은 출력
# ============================================================

def print_result(
    name: str,
    result: dict,
):

    print(
        f"{name}"
    )

    print(
        f"  Overall MAE : "
        f"{result['overall_mae_cm']:.4f} cm"
    )

    print(
        f"  Overall RMSE: "
        f"{result['overall_rmse_cm']:.4f} cm"
    )

    for horizon in (
        5,
        10,
        15,
    ):
        index = horizon - 1

        print(
            f"  +{horizon:2d} min "
            f"MAE={result['horizon_mae_cm'][index]:.4f} cm, "
            f"RMSE={result['horizon_rmse_cm'][index]:.4f} cm, "
            f"N={result['observed_count'][index]}"
        )


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
        "--output",
        type=Path,
        default=Path(
            "artifacts/baselines/"
            "baseline_results.json"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Development data
    # --------------------------------------------------------

    data = load_development_data(
        args.data_dir
    )

    scaler = load_scaler(
        args.data_dir
    )

    results = []

    # --------------------------------------------------------
    # 4 CV folds
    # --------------------------------------------------------

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

        print(
            "\n"
            "================================"
        )

        print(
            f"FOLD {fold_number}"
        )

        print(
            "Validation samples:",
            len(val_indices),
        )

        print(
            "================================"
        )

        # ----------------------------------------------------
        # Persistence
        # ----------------------------------------------------

        persistence_result = (
            evaluate_model(
                model_fn=(
                    persistence_forecast
                ),
                data=data,
                indices=val_indices,
                scaler=scaler,
                batch_size=args.batch_size,
            )
        )

        print_result(
            "Persistence",
            persistence_result,
        )

        # ----------------------------------------------------
        # Linear 5-min
        # ----------------------------------------------------

        linear_result = (
            evaluate_model(
                model_fn=(
                    lambda X:
                    linear_forecast(
                        X,
                        history=5,
                    )
                ),
                data=data,
                indices=val_indices,
                scaler=scaler,
                batch_size=args.batch_size,
            )
        )

        print_result(
            "Linear-5min",
            linear_result,
        )

        results.append(
            {
                "fold": fold_number,
                "validation_samples":
                    int(
                        len(
                            val_indices
                        )
                    ),

                "persistence":
                    persistence_result,

                "linear_5min":
                    linear_result,
            }
        )

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

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
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # CV 평균 / 표준편차
    # --------------------------------------------------------

    persistence_mae = np.array(
        [
            fold["persistence"]
            ["overall_mae_cm"]
            for fold in results
        ]
    )

    linear_mae = np.array(
        [
            fold["linear_5min"]
            ["overall_mae_cm"]
            for fold in results
        ]
    )

    persistence_rmse = np.array(
        [
            fold["persistence"]
            ["overall_rmse_cm"]
            for fold in results
        ]
    )

    linear_rmse = np.array(
        [
            fold["linear_5min"]
            ["overall_rmse_cm"]
            for fold in results
        ]
    )

    print(
        "\n"
        "================================"
    )

    print(
        "CV SUMMARY"
    )

    print(
        "================================"
    )

    print(
        "Persistence MAE : "
        f"{persistence_mae.mean():.4f} "
        f"± {persistence_mae.std(ddof=1):.4f} cm"
    )

    print(
        "Persistence RMSE: "
        f"{persistence_rmse.mean():.4f} "
        f"± {persistence_rmse.std(ddof=1):.4f} cm"
    )

    print(
        "Linear MAE      : "
        f"{linear_mae.mean():.4f} "
        f"± {linear_mae.std(ddof=1):.4f} cm"
    )

    print(
        "Linear RMSE     : "
        f"{linear_rmse.mean():.4f} "
        f"± {linear_rmse.std(ddof=1):.4f} cm"
    )

    print(
        "\nBaseline evaluation passed."
    )


if __name__ == "__main__":
    main()