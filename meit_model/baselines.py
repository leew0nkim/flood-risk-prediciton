import torch


def persistence_forecast(
    X: torch.Tensor,
    horizon: int = 15,
) -> torch.Tensor:
    """
    마지막 관측 수위가 향후에도 그대로 유지된다고 가정한다.

    X:
        (B, 30, 1)

    return:
        (B, 15, 1)
    """

    last_value = X[:, -1:, :]

    return last_value.repeat(
        1,
        horizon,
        1,
    )


def linear_forecast(
    X: torch.Tensor,
    history: int = 5,
    horizon: int = 15,
) -> torch.Tensor:
    """
    최근 history분의 수위에 직선을 fitting하고
    향후 1~15분 수위를 선형 외삽한다.

    X:
        (B, 30, 1)

    return:
        (B, 15, 1)
    """

    if X.ndim != 3 or X.shape[-1] != 1:
        raise ValueError(
            "X must have shape (B, history_length, 1)"
        )

    if X.shape[1] < history:
        raise ValueError(
            "Not enough history for linear forecast"
        )

    recent = X[:, -history:, 0]

    device = X.device
    dtype = X.dtype

    # 최근 5개 시점을 0,1,2,3,4로 둔다.
    t = torch.arange(
        history,
        dtype=dtype,
        device=device,
    )

    t_mean = t.mean()
    t_centered = t - t_mean

    y_mean = recent.mean(
        dim=1,
        keepdim=True,
    )

    y_centered = (
        recent - y_mean
    )

    denominator = torch.sum(
        t_centered.square()
    )

    slope = torch.sum(
        y_centered
        * t_centered,
        dim=1,
    ) / denominator

    intercept = (
        y_mean.squeeze(1)
        - slope * t_mean
    )

    # history=5이면 입력 시점은 0~4.
    # 미래 +1분부터는 5~19.
    future_t = torch.arange(
        history,
        history + horizon,
        dtype=dtype,
        device=device,
    )

    prediction = (
        intercept[:, None]
        + slope[:, None]
        * future_t[None, :]
    )

    return prediction.unsqueeze(-1)