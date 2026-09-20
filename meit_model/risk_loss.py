import torch


def risk_aware_masked_mae(
    prediction,
    target,
    observed,
    scaler,
    critical_depth_m=0.30,
    underprediction_weight=2.0,
):
    """
    위험수위에서 실제보다 낮게 예측한 경우
    더 큰 가중치를 주는 asymmetric MAE.

    prediction, target:
        standardized depth
        shape = (B, 15, 1)

    observed:
        bool mask
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes must match"
        )

    if observed.shape != target.shape:
        raise ValueError(
            "Observation mask shape must match target"
        )

    if observed.dtype != torch.bool:
        raise TypeError(
            "Observation mask must have boolean dtype"
        )

    valid = (
        observed
        & torch.isfinite(target)
    )

    if not valid.any():
        raise ValueError(
            "No observed finite labels in this batch"
        )

    # NaN target과 먼저 연산하면 안 되므로
    # valid 값만 먼저 선택한다.
    prediction_valid = prediction[valid]
    target_valid = target[valid]

    if not torch.isfinite(
        prediction_valid
    ).all():
        raise ValueError(
            "Nonfinite prediction at an observed target"
        )

    # --------------------------------------------------------
    # 실제 target을 metre 단위로 복원
    # risk mask를 만들기 위한 용도
    # --------------------------------------------------------

    target_m = (
        target_valid
        * scaler["std_m"]
        + scaler["mean_m"]
    )

    # --------------------------------------------------------
    # 기본 standardized MAE
    # --------------------------------------------------------

    absolute_error = torch.abs(
        prediction_valid
        - target_valid
    )

    # --------------------------------------------------------
    # 위험구간 underprediction 판정
    #
    # actual >= 0.30m
    # AND
    # prediction < actual
    # --------------------------------------------------------

    risk_underprediction = (
        (target_m >= critical_depth_m)
        & (
            prediction_valid
            < target_valid
        )
    )

    weights = torch.ones_like(
        absolute_error
    )

    weights = torch.where(
        risk_underprediction,
        torch.full_like(
            weights,
            underprediction_weight,
        ),
        weights,
    )

    # weighted mean
    return (
        absolute_error
        * weights
    ).sum() / weights.sum()
