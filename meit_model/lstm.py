import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """
    최근 30분 외부 수위로
    향후 1~15분 외부 수위를 한 번에 예측한다.

    Input:
        (B, 30, 1)

    Output:
        (B, 15, 1)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        horizon: int = 15,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            horizon,
        )

    def forward(self, x):
        # x: (B, 30, 1)

        _, (hidden, _) = self.lstm(x)

        # hidden:
        # (num_layers, B, hidden_size)

        last_hidden = hidden[-1]

        # (B, 15)
        prediction = self.output_layer(
            last_hidden
        )

        # (B, 15, 1)
        return prediction.unsqueeze(-1)
