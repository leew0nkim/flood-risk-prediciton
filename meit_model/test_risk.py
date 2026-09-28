from datetime import datetime, timedelta
import math

from meit_model.predictor import ForecastResult
from meit_model.risk import (
    RiskEngine,
    SAFE,
    WARNING,
    DANGER,
)


BASE_TIME = datetime(2026, 9, 23, 9, 0)


def make_forecast(
    values=None,
    status="READY",
):
    if status == "READY":
        if values is None:
            values = [0.20] * 15

        forecast_times = [
            BASE_TIME + timedelta(minutes=i)
            for i in range(1, 16)
        ]

        return ForecastResult(
            status="READY",
            predicted_depths_m=values,
            forecast_times=forecast_times,
        )

    return ForecastResult(
        status=status,
        predicted_depths_m=None,
        forecast_times=[],
    )


def test_safe():
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.20,
        internal_depth_m=0.05,
        equalization_active=False,
        forecast=make_forecast(
            [0.22] * 15
        ),
    )

    assert result.external_state == SAFE
    assert result.internal_state == SAFE
    assert result.current_state == SAFE

    assert result.t_critical_min is None
    assert result.forecast_state == SAFE
    assert result.state == SAFE

    assert result.pressure_direction == "OUTSIDE_HIGHER"
    assert (
        result.pressure_forecast_status
        == "CONDITIONAL_INTERNAL_CONSTANT"
    )


def test_external_warning():
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.25,
        internal_depth_m=0.05,
        equalization_active=False,
        forecast=make_forecast(
            [0.22] * 15
        ),
    )

    assert result.external_state == WARNING
    assert result.internal_state == SAFE
    assert result.current_state == WARNING
    assert result.state == WARNING


def test_internal_warning():
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.10,
        internal_depth_m=0.25,
        equalization_active=False,
        forecast=make_forecast(
            [0.20] * 15
        ),
    )

    assert result.external_state == SAFE
    assert result.internal_state == WARNING
    assert result.current_state == WARNING
    assert result.state == WARNING

    assert result.pressure_direction == "INSIDE_HIGHER"


def test_external_danger():
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.30,
        internal_depth_m=0.05,
        equalization_active=False,
        forecast=make_forecast(
            [0.20] * 15
        ),
    )

    assert result.external_state == DANGER
    assert result.current_state == DANGER
    assert result.state == DANGER

    # 현재 이미 외부 critical 도달
    assert result.t_external_critical_min == 0

    # 현재 0.29 warning도 이미 지난 상태
    assert result.t_external_warning_min == 0


def test_current_external_warning_time_zero():
    """
    현재 외부수위가 이미 0.29m 이상이면,
    미래 예측과 관계없이 external warning time은 0분이어야 한다.
    """
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.295,
        internal_depth_m=0.05,
        equalization_active=False,
        forecast=make_forecast(
            [0.20] * 15
        ),
    )

    assert result.t_external_warning_min == 0
    assert result.t_external_critical_min is None

    assert result.current_state == WARNING
    assert result.state == WARNING


def test_internal_danger():
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.10,
        internal_depth_m=0.30,
        equalization_active=False,
        forecast=make_forecast(
            [0.20] * 15
        ),
    )

    assert result.external_state == SAFE
    assert result.internal_state == DANGER
    assert result.current_state == DANGER
    assert result.state == DANGER


def test_future_pressure_warning():
    """
    현재는 안전하지만,
    미래 등가수심이 10분 뒤 0.30m에 도달.
    → LED WARNING
    """
    engine = RiskEngine()

    predictions = (
        [0.29] * 9
        + [0.31] * 6
    )

    result = engine.evaluate(
        current_depth_m=0.20,
        internal_depth_m=0.05,
        equalization_active=False,
        forecast=make_forecast(
            predictions
        ),
    )

    assert result.current_state == SAFE

    assert result.t_critical_min == 10
    assert result.forecast_state == WARNING

    assert result.state == WARNING


def test_future_pressure_danger():
    """
    미래 등가수심 critical까지 4분.
    danger horizon=5분이므로 DANGER.
    """
    engine = RiskEngine(
        danger_horizon_min=5
    )

    predictions = (
        [0.29] * 3
        + [0.31] * 12
    )

    result = engine.evaluate(
        current_depth_m=0.20,
        internal_depth_m=0.05,
        equalization_active=False,
        forecast=make_forecast(
            predictions
        ),
    )

    assert result.current_state == SAFE

    assert result.t_critical_min == 4
    assert result.forecast_state == DANGER

    assert result.state == DANGER


def test_external_forecast_warning_is_not_led_rule():
    """
    외부수위 예측이 0.29m를 넘었다는 것만으로
    LED WARNING이 되어서는 안 된다.

    LED 미래 위험은 미래 등가수심 기반 T_critical로 판단한다.
    """
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.20,
        internal_depth_m=0.20,
        equalization_active=False,
        forecast=make_forecast(
            [0.29] * 15
        ),
    )

    # 외부 예측 diagnostic은 crossing
    assert result.t_external_warning_min == 1

    # 하지만 등가수심 critical은 아님
    assert result.t_critical_min is None

    assert result.current_state == SAFE
    assert result.forecast_state == SAFE
    assert result.state == SAFE


def test_high_but_balanced_water_is_flood_danger():
    """
    실내외 모두 0.50m라 압력차는 거의 없더라도,
    침수 자체는 위험하므로 LED는 DANGER.
    """
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.50,
        internal_depth_m=0.50,
        equalization_active=False,
        forecast=make_forecast(
            [0.50] * 15
        ),
    )

    assert result.current_state == DANGER
    assert result.state == DANGER

    assert result.pressure_direction == "BALANCED"

    assert math.isclose(
        result.current_equivalent_depth_m,
        0.0,
        abs_tol=1e-12,
    )

    assert result.t_critical_min is None


def test_current_pressure_already_critical():
    """
    현재 등가수심이 이미 0.30m critical이면
    미래 LSTM이 아직 WARMUP이어도 T_critical=0이어야 한다.

    outside=0.50, inside=0.40
    sqrt(0.50^2 - 0.40^2) = 0.30
    """
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.50,
        internal_depth_m=0.40,
        equalization_active=False,
        forecast=make_forecast(
            status="WARMUP"
        ),
    )

    assert result.pressure_direction == "OUTSIDE_HIGHER"

    assert math.isclose(
        result.current_equivalent_depth_m,
        0.30,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert result.t_critical_min == 0

    # 현재 침수상태도 이미 위험
    assert result.state == DANGER


def test_warmup_is_warning_when_currently_safe():
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.20,
        internal_depth_m=0.05,
        equalization_active=False,
        forecast=make_forecast(
            status="WARMUP"
        ),
    )

    assert result.current_state == SAFE

    # 미래 압력 상태를 확정할 수 없으므로 보수적으로 WARNING
    assert result.forecast_state == WARNING
    assert result.state == WARNING


def test_equalization_active_disables_future_pressure_estimate():
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.20,
        internal_depth_m=0.05,
        equalization_active=True,
        forecast=make_forecast(
            [0.35] * 15
        ),
    )

    assert (
        result.pressure_forecast_status
        == "UNAVAILABLE_EQUALIZATION_ACTIVE"
    )

    assert result.t_critical_min is None
    assert result.forecast_state == WARNING


def test_missing_internal_depth_is_warning():
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.20,
        internal_depth_m=None,
        equalization_active=False,
        forecast=make_forecast(
            [0.20] * 15
        ),
    )

    assert (
        result.pressure_status
        == "MISSING_INTERNAL_DEPTH"
    )

    # 내부 정보를 모르면 SAFE라고 단정하지 않음
    assert result.state == WARNING


def test_invalid_equalization_active():
    engine = RiskEngine()

    try:
        engine.evaluate(
            current_depth_m=0.20,
            internal_depth_m=0.05,
            equalization_active="false",
            forecast=make_forecast(),
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "String 'false' must not be accepted as bool"
        )


def test_invalid_forecast_length():
    engine = RiskEngine()

    result = engine.evaluate(
        current_depth_m=0.20,
        internal_depth_m=0.05,
        equalization_active=False,
        forecast=make_forecast(
            [0.20] * 14
        ),
    )

    assert result.forecast_status == "INVALID_FORECAST"
    assert result.forecast_state == WARNING
    assert result.state == WARNING


def main():

    tests = [
        test_safe,
        test_external_warning,
        test_internal_warning,
        test_external_danger,
        test_current_external_warning_time_zero,
        test_internal_danger,
        test_future_pressure_warning,
        test_future_pressure_danger,
        test_external_forecast_warning_is_not_led_rule,
        test_high_but_balanced_water_is_flood_danger,
        test_current_pressure_already_critical,
        test_warmup_is_warning_when_currently_safe,
        test_equalization_active_disables_future_pressure_estimate,
        test_missing_internal_depth_is_warning,
        test_invalid_equalization_active,
        test_invalid_forecast_length,
    ]

    for test in tests:
        test()
        print(
            f"[PASS] {test.__name__}"
        )

    print()
    print("=" * 40)
    print(
        f"RISK TEST PASSED: "
        f"{len(tests)}/{len(tests)}"
    )
    print("=" * 40)


if __name__ == "__main__":
    main()
