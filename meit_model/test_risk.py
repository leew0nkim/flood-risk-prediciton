from meit_model.predictor import ForecastResult
from meit_model.risk import RiskEngine


engine = RiskEngine(
    warning_current_depth_m=0.25,
    external_critical_depth_m=0.30,
    forecast_warning_depth_m=0.29,
    danger_horizon_min=5,
)


def make_forecast(values):
    return ForecastResult(
        status="READY",
        predicted_depths_m=values,
        forecast_times=None,
    )


# ----------------------------------
# TEST 1: SAFE
# ----------------------------------
forecast = make_forecast(
    [0.20 + i * 0.001 for i in range(15)]
)

result = engine.evaluate(
    current_depth_m=0.20,
    forecast=forecast,
)

print("\nTEST 1")
print(result)


# ----------------------------------
# TEST 2: 현재 수위 WARNING
# ----------------------------------
forecast = make_forecast(
    [0.26] * 15
)

result = engine.evaluate(
    current_depth_m=0.26,
    forecast=forecast,
)

print("\nTEST 2")
print(result)


# ----------------------------------
# TEST 3: 미래 10분 후 warning
# → WARNING
# ----------------------------------
values = [0.27] * 9 + [0.291] * 6

result = engine.evaluate(
    current_depth_m=0.22,
    forecast=make_forecast(values),
)

print("\nTEST 3")
print(result)


# ----------------------------------
# TEST 4: 미래 4분 후 warning
# → DANGER
# ----------------------------------
values = [
    0.27,
    0.275,
    0.280,
    0.291,
] + [0.295] * 11

result = engine.evaluate(
    current_depth_m=0.22,
    forecast=make_forecast(values),
)

assert result.state == "DANGER"
assert result.t_warning_min == 4
assert result.t_external_critical_min is None
assert not hasattr(result, "command")
print("\nTEST 4")
print(result)


# ----------------------------------
# TEST 5: 현재 이미 critical
# → DANGER
# ----------------------------------
forecast = make_forecast(
    [0.31] * 15
)

result = engine.evaluate(
    current_depth_m=0.305,
    forecast=forecast,
)

print("\nTEST 5")
print(result)


# ----------------------------------
# TEST 6: WARMUP
# → SAFE로 처리하면 안 됨
# ----------------------------------
forecast = ForecastResult(
    status="WARMUP",
    predicted_depths_m=None,
    forecast_times=None,
    message="10/30분 데이터 수집 중",
)

result = engine.evaluate(
    current_depth_m=0.20,
    forecast=forecast,
)

print("\nTEST 6")
print(result)