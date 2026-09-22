from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import math
import torch

from meit_model.lstm import LSTMForecaster


INPUT_MINUTES = 30
FORECAST_MINUTES = 15


@dataclass
class ForecastResult:
    status: str
    predicted_depths_m: Optional[list[float]]
    forecast_times: Optional[list[datetime]]
    message: Optional[str] = None


class ForecastPredictor:
    """
    최근 30분 외부수위 -> 미래 15분 외부수위 예측.

    내부 단위는 항상 meter.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        # -----------------------------------------------
        # 1. scaler
        # -----------------------------------------------
        if "scaler" not in checkpoint:
            raise KeyError(
                "Checkpoint에 scaler가 없습니다. "
                "fold-specific scaler가 저장된 checkpoint를 사용하세요."
            )

        scaler = checkpoint["scaler"]

        self.mean_m = float(scaler["mean_m"])
        self.std_m = float(scaler["std_m"])

        if not math.isfinite(self.mean_m):
            raise ValueError("scaler mean_m이 유효하지 않습니다.")

        if not math.isfinite(self.std_m) or self.std_m <= 0:
            raise ValueError("scaler std_m이 유효하지 않습니다.")

        # -----------------------------------------------
        # 2. model
        # -----------------------------------------------
        config = checkpoint.get("config", {})
        input_size = config.get("input_size", 1)
        horizon = config.get("horizon", 15)
        if input_size != 1 or horizon != FORECAST_MINUTES:
            raise ValueError("Predictor requires input_size=1 and horizon=15")
        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 1),
            horizon=horizon,
        ).to(self.device)

        state_dict = self._extract_state_dict(checkpoint)
        self.model.load_state_dict(state_dict)

        self.model.eval()

        print(f"Predictor loaded: {self.checkpoint_path}")
        print(f"Device: {self.device}")
        print(
            f"Scaler: mean={self.mean_m:.6f} m, "
            f"std={self.std_m:.6f} m"
        )

    @staticmethod
    def _extract_state_dict(checkpoint: dict):
        """
        train script의 checkpoint key 이름 차이를 조금 유연하게 처리.
        """
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        if "model" in checkpoint:
            return checkpoint["model"]

        raise KeyError(
            "Checkpoint에서 model state dict를 찾을 수 없습니다. "
            f"keys={list(checkpoint.keys())}"
        )

    def _validate_input(
        self,
        recent_depths_m: Sequence[float],
    ) -> list[float]:
        if len(recent_depths_m) < INPUT_MINUTES:
            raise ValueError(
                f"최근 {INPUT_MINUTES}분 데이터가 필요합니다. "
                f"현재 {len(recent_depths_m)}개입니다."
            )

        # 혹시 더 많이 들어오면 최신 30분만 사용
        values = list(recent_depths_m)[-INPUT_MINUTES:]

        clean_values = []

        for value in values:
            if value is None:
                raise ValueError("입력 수위에 None이 포함되어 있습니다.")

            value = float(value)

            if not math.isfinite(value):
                raise ValueError(
                    "입력 수위에 NaN/inf가 포함되어 있습니다."
                )

            if value < 0:
                raise ValueError(
                    f"음수 수위가 입력되었습니다: {value}"
                )

            clean_values.append(value)

        return clean_values

    def predict(
        self,
        recent_depths_m: Sequence[float],
        *,
        origin_time: Optional[datetime] = None,
    ) -> ForecastResult:
        """
        최근 30분 수위로 미래 15분 수위를 예측한다.

        Parameters
        ----------
        recent_depths_m:
            meter 단위 외부수위. 최소 30개.

        origin_time:
            마지막 관측 시각.
            없으면 현재 시각 사용.
        """

        try:
            values = self._validate_input(recent_depths_m)
        except (ValueError, TypeError, OverflowError) as exc:
            return ForecastResult(
                status="INSUFFICIENT_DATA",
                predicted_depths_m=None,
                forecast_times=None,
                message=str(exc),
            )

        if origin_time is None:
            origin_time = datetime.now()

        # -----------------------------------------------
        # input normalization
        # -----------------------------------------------
        x = torch.tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

        x = (x - self.mean_m) / self.std_m

        if not torch.isfinite(x).all():
            return ForecastResult("INSUFFICIENT_DATA", None, None, "Nonfinite normalized input")

        # (30,) -> (1, 30, 1)
        x = x.view(1, INPUT_MINUTES, 1)

        # -----------------------------------------------
        # inference
        # -----------------------------------------------
        with torch.no_grad():
            prediction_std = self.model(x)

        # 예상 shape: (1, 15, 1)
        if prediction_std.shape != (1, FORECAST_MINUTES, 1):
            return ForecastResult("PREDICTION_ERROR", None, None,
                                  f"Unexpected output shape: {tuple(prediction_std.shape)}")

        # -----------------------------------------------
        # inverse normalization
        # -----------------------------------------------
        prediction_m = (
            prediction_std * self.std_m + self.mean_m
        )

        # 실제 수위는 음수가 될 수 없으므로 reporting prediction만 clip
        if not torch.isfinite(prediction_m).all():
            return ForecastResult("PREDICTION_ERROR", None, None, "Nonfinite model prediction")

        prediction_m = torch.clamp_min(
            prediction_m,
            0.0,
        )

        predictions = (
            prediction_m
            .squeeze(0)
            .squeeze(-1)
            .cpu()
            .tolist()
        )

        forecast_times = [
            origin_time + timedelta(minutes=i)
            for i in range(1, FORECAST_MINUTES + 1)
        ]

        return ForecastResult(
            status="READY",
            predicted_depths_m=predictions,
            forecast_times=forecast_times,
        )


class LiveForecastBuffer:
    """Buffer consecutive completed one-minute bins, NOT raw sensor packets.

    timestamp must be the bin-end time on a fixed 60-second grid. The caller
    supplies each completed bin's representative depth (metres). This class
    does not aggregate high-frequency samples or forward-fill missing bins.
    Omitting timestamp uses receipt time; explicit bin-end timestamps are
    recommended because clock/network jitter is not a valid bin definition.
    """

    def __init__(self, predictor: ForecastPredictor):
        self.predictor = predictor
        self.buffer = deque(maxlen=INPUT_MINUTES)
        self.timestamps = deque(maxlen=INPUT_MINUTES)
        self._last_timestamp = None

    @property
    def size(self):
        return len(self.buffer)

    @property
    def ready(self):
        """History completeness only; each new input must still pass validation."""
        return self.size == INPUT_MINUTES

    def reset(self):
        self.buffer.clear()
        self.timestamps.clear()
        self._last_timestamp = None

    def add_measurement(self, depth_m, *, timestamp=None):
        timestamp = datetime.now() if timestamp is None else timestamp
        if not isinstance(timestamp, datetime):
            return ForecastResult("INSUFFICIENT_DATA", None, None, "Invalid timestamp")
        delta = None
        if self._last_timestamp is not None:
            try:
                delta = (timestamp - self._last_timestamp).total_seconds()
            except TypeError:
                return ForecastResult("INSUFFICIENT_DATA", None, None,
                                      "Do not mix timezone-aware and naive timestamps")
            if delta <= 0:
                return ForecastResult("INSUFFICIENT_DATA", None, None,
                                      "Duplicate or out-of-order timestamp; input ignored")
        try:
            value = float(depth_m)
            valid = math.isfinite(value) and value >= 0
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            self.reset()
            self._last_timestamp = timestamp
            return ForecastResult("INSUFFICIENT_DATA", None, None,
                                  "Missing/invalid depth; history cleared")
        if delta is not None and delta < 60:
            return ForecastResult("INSUFFICIENT_DATA", None, None,
                                  "Expected one completed bin per 60 seconds; input ignored")
        gap = delta is not None and delta > 60
        if gap:
            self.reset()
        self.buffer.append(value)
        self.timestamps.append(timestamp)
        self._last_timestamp = timestamp
        if not self.ready:
            prefix = "Time gap; history restarted. " if gap else ""
            return ForecastResult("WARMUP", None, None,
                                  f"{prefix}{self.size}/30 consecutive minute bins")
        return self.predictor.predict(list(self.buffer), origin_time=timestamp)
