"""External-depth early warning for LED/voice only.

DANGER means a red early warning, not physical inability to open a door.
This module issues no OPEN/CLOSE command. A separate controller must evaluate
indoor/outdoor depths, equivalent depth and evacuation status.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import math

from meit_model.predictor import ForecastResult


SAFE = "SAFE"
WARNING = "WARNING"
DANGER = "DANGER"


@dataclass
class RiskResult:
    # LED/voice state only; never a valve actuation command.
    state: str

    # 현재 수위
    current_depth_m: float

    # 예측 기준
    warning_depth_m: float
    external_critical_depth_m: float

    # calibrated warning threshold 최초 도달 예상시간
    t_warning_min: Optional[int]

    # 실제 critical depth(0.30m) 최초 도달 예상시간
    # External-depth reference only, NOT the physical door-opening T_critical.
    t_external_critical_min: Optional[int]

    # predictor 상태
    forecast_status: str

    # 사람이 확인하기 위한 설명
    reason: str


class RiskEngine:
    """
    수위 예측 결과를 SAFE / WARNING / DANGER 상태로 변환한다.

    내부 단위는 모두 meter.

    기본 정책
    ----------
    SAFE:
        현재 수위 < 0.25m
        AND 앞으로 15분 동안 warning threshold 도달 예측 없음

    WARNING:
        현재 수위가 0.25m 이상 0.30m 미만
        OR warning threshold 도달까지 6~15분
        OR 예측 불가능(WARMUP / INSUFFICIENT_DATA)

    DANGER:
        현재 수위 >= 0.30m
        OR warning threshold 도달까지 5분 이하
    """

    def __init__(
        self,
        *,
        warning_current_depth_m: float = 0.25,
        external_critical_depth_m: float = 0.30,
        forecast_warning_depth_m: float = 0.29,
        danger_horizon_min: int = 5,
    ):
        if not (
            math.isfinite(external_critical_depth_m)
            and 0 <= warning_current_depth_m
            < external_critical_depth_m
        ):
            raise ValueError(
                "warning_current_depth_m은 "
                "external_critical_depth_m보다 작아야 합니다."
            )

        if not (math.isfinite(forecast_warning_depth_m)
                and 0 < forecast_warning_depth_m <= external_critical_depth_m):
            raise ValueError(
                "forecast_warning_depth_m은 양수여야 합니다."
            )

        if (not math.isfinite(danger_horizon_min)
                or int(danger_horizon_min) != danger_horizon_min
                or not 1 <= danger_horizon_min <= 15):
            raise ValueError(
                "danger_horizon_min은 양수여야 합니다."
            )

        self.warning_current_depth_m = (
            float(warning_current_depth_m)
        )

        self.external_critical_depth_m = float(
            external_critical_depth_m
        )

        self.forecast_warning_depth_m = float(
            forecast_warning_depth_m
        )

        self.danger_horizon_min = int(
            danger_horizon_min
        )

    @staticmethod
    def _validate_current_depth(
        current_depth_m: float,
    ) -> float:
        value = float(current_depth_m)

        if not math.isfinite(value):
            raise ValueError(
                "현재 수위가 NaN/inf입니다."
            )

        if value < 0:
            raise ValueError(
                "현재 수위가 음수입니다."
            )

        return value

    @staticmethod
    def _first_crossing_min(
        predictions_m: Sequence[float],
        threshold_m: float,
    ) -> Optional[int]:
        """
        미래 +1 ~ +15분 예측 중 threshold에
        처음 도달하는 시점을 반환한다.

        없으면 None.
        """

        for minute, depth in enumerate(
            predictions_m,
            start=1,
        ):
            depth = float(depth)

            if not math.isfinite(depth):
                continue

            if depth >= threshold_m:
                return minute

        return None

    def evaluate(
        self,
        current_depth_m: float,
        forecast: ForecastResult,
    ) -> RiskResult:

        current_depth_m = (
            self._validate_current_depth(
                current_depth_m
            )
        )

        # --------------------------------------------------
        # 1. 현재 이미 critical이면 예측과 상관없이 DANGER
        # --------------------------------------------------
        if current_depth_m >= self.external_critical_depth_m:
            return RiskResult(
                state=DANGER,
                current_depth_m=current_depth_m,
                warning_depth_m=(
                    self.forecast_warning_depth_m
                ),
                external_critical_depth_m=self.external_critical_depth_m,
                t_warning_min=0,
                t_external_critical_min=0,
                forecast_status=forecast.status,
                reason=(
                    "현재 수위가 critical depth "
                    f"{self.external_critical_depth_m:.3f}m "
                    "이상입니다."
                ),
            )

        # --------------------------------------------------
        # 2. 예측 불가능
        #
        # 예측 데이터가 부족하다고 SAFE로 판정하면 안 된다.
        # 1차 시스템에서는 보수적으로 WARNING 처리.
        # --------------------------------------------------
        if (
            forecast.status != "READY"
            or forecast.predicted_depths_m is None
        ):
            return RiskResult(
                state=WARNING,
                current_depth_m=current_depth_m,
                warning_depth_m=(
                    self.forecast_warning_depth_m
                ),
                external_critical_depth_m=self.external_critical_depth_m,
                t_warning_min=None,
                t_external_critical_min=None,
                forecast_status=forecast.status,
                reason=(
                    "미래 예측이 준비되지 않아 "
                    "보수적으로 WARNING 처리합니다."
                ),
            )

        predictions = forecast.predicted_depths_m
        try:
            valid_predictions = len(predictions) == 15 and all(
                math.isfinite(float(x)) and float(x) >= 0 for x in predictions)
        except (TypeError, ValueError, OverflowError):
            valid_predictions = False
        if not valid_predictions:
            return RiskResult(
                state=WARNING, current_depth_m=current_depth_m,
                warning_depth_m=self.forecast_warning_depth_m,
                external_critical_depth_m=self.external_critical_depth_m,
                t_warning_min=None, t_external_critical_min=None,
                forecast_status="INVALID_FORECAST",
                reason="Invalid 15-minute forecast; warning only, no actuation.",
            )

        # --------------------------------------------------
        # 3. 최초 warning / critical crossing 계산
        # --------------------------------------------------
        t_warning = self._first_crossing_min(
            predictions,
            self.forecast_warning_depth_m,
        )

        t_external_critical = self._first_crossing_min(
            predictions,
            self.external_critical_depth_m,
        )

        # --------------------------------------------------
        # 4. 미래 위험이 5분 이내
        # --------------------------------------------------
        if (
            t_warning is not None
            and t_warning <= self.danger_horizon_min
        ):
            return RiskResult(
                state=DANGER,
                current_depth_m=current_depth_m,
                warning_depth_m=(
                    self.forecast_warning_depth_m
                ),
                external_critical_depth_m=self.external_critical_depth_m,
                t_warning_min=t_warning,
                t_external_critical_min=t_external_critical,
                forecast_status=forecast.status,
                reason=(
                    "예측 경고 수위 "
                    f"{self.forecast_warning_depth_m:.3f}m "
                    f"도달까지 {t_warning}분으로 "
                    "DANGER 기준 이내입니다."
                ),
            )

        # --------------------------------------------------
        # 5. 미래 위험이 6~15분 이내
        # --------------------------------------------------
        if t_warning is not None:
            return RiskResult(
                state=WARNING,
                current_depth_m=current_depth_m,
                warning_depth_m=(
                    self.forecast_warning_depth_m
                ),
                external_critical_depth_m=self.external_critical_depth_m,
                t_warning_min=t_warning,
                t_external_critical_min=t_external_critical,
                forecast_status=forecast.status,
                reason=(
                    "예측 경고 수위 "
                    f"{self.forecast_warning_depth_m:.3f}m "
                    f"도달까지 {t_warning}분 남았습니다."
                ),
            )

        # --------------------------------------------------
        # 6. 미래 crossing은 없지만 현재 수위가 warning 영역
        # --------------------------------------------------
        if (
            current_depth_m
            >= self.warning_current_depth_m
        ):
            return RiskResult(
                state=WARNING,
                current_depth_m=current_depth_m,
                warning_depth_m=(
                    self.forecast_warning_depth_m
                ),
                external_critical_depth_m=self.external_critical_depth_m,
                t_warning_min=None,
                t_external_critical_min=None,
                forecast_status=forecast.status,
                reason=(
                    "15분 내 위험 도달 예측은 없지만 "
                    "현재 수위가 WARNING 기준 "
                    f"{self.warning_current_depth_m:.3f}m "
                    "이상입니다."
                ),
            )

        # --------------------------------------------------
        # 7. 현재도 낮고 미래 위험도 없음
        # --------------------------------------------------
        return RiskResult(
            state=SAFE,
            current_depth_m=current_depth_m,
            warning_depth_m=(
                self.forecast_warning_depth_m
            ),
            external_critical_depth_m=self.external_critical_depth_m,
            t_warning_min=None,
            t_external_critical_min=None,
            forecast_status=forecast.status,
            reason=(
                "현재 수위가 낮고 "
                "15분 내 위험 도달 예측이 없습니다."
            ),
        )