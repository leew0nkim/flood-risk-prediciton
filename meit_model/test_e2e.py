from datetime import datetime, timedelta

from meit_model.predictor import (
    ForecastPredictor,
    LiveForecastBuffer,
)

from meit_model.risk import (
    RiskEngine,
    DANGER,
)

from meit_model.controller import (
    SafetyController,
    OPEN,
    CLOSE,
)


CHECKPOINT = (
    "artifacts/risk_lstm_cv_v2/"
    "fold_1/best_model.pt"
)


# ==================================================
# 1. 시스템 구성
# ==================================================

predictor = ForecastPredictor(CHECKPOINT)
live = LiveForecastBuffer(predictor)

risk_engine = RiskEngine(
    warning_current_depth_m=0.25,
    external_critical_depth_m=0.30,
    forecast_warning_depth_m=0.29,
    danger_horizon_min=5,
)

controller = SafetyController()


# ==================================================
# 2. Fake sensor data
#
# 1~30분:
#   0.20m로 안정 → predictor warmup
#
# 이후:
#   천천히 수위 상승
# ==================================================

fake_depths_m = (
    [0.20] * 30
    + [
        0.205,
        0.210,
        0.215,
        0.220,
        0.225,
        0.230,
        0.235,
        0.240,
        0.245,
        0.250,
        0.255,
        0.260,
        0.265,
        0.270,
        0.275,
        0.280,
        0.285,
        0.290,
        0.295,
        0.300,
        0.305,
    ]
)


start_time = datetime(
    2026,
    9,
    23,
    9,
    0,
)


# ==================================================
# 3. simulation state
# ==================================================

equalization_active = False

payloads = []

open_count = 0
close_count = 0


# ==================================================
# 4. E2E simulation
# ==================================================

for index, depth_m in enumerate(fake_depths_m):

    minute = index + 1

    timestamp = (
        start_time
        + timedelta(minutes=index)
    )

    # ----------------------------------------------
    # Sensor → Predictor
    # ----------------------------------------------

    forecast = live.add_measurement(
        depth_m,
        timestamp=timestamp,
    )

    # ----------------------------------------------
    # Predictor → Risk
    # ----------------------------------------------

    risk = risk_engine.evaluate(
        current_depth_m=depth_m,
        forecast=forecast,
    )

    # ----------------------------------------------
    # MOCK physical decision
    #
    # 실제 등가수심 로직이 아니다.
    #
    # 43~48분:
    #   압력평형 필요하다고 가정
    #
    # 49분 이후:
    #   대피 완료 가정
    # ----------------------------------------------

    equalization_required = (
        43 <= minute < 49
    )

    equalization_allowed = True

    evacuation_complete = (
        minute >= 49
    )

    # ----------------------------------------------
    # Risk + mock physical state → Controller
    # ----------------------------------------------

    control = controller.decide(
        risk,
        evacuation_complete=evacuation_complete,
        equalization_required=equalization_required,
        equalization_allowed=equalization_allowed,
        equalization_active=equalization_active,
    )

    # ----------------------------------------------
    # Fake hardware feedback
    #
    # 실제 연동에서는 command 전송만으로
    # active=True로 바꾸면 안 된다.
    #
    # 여기서는 fake hardware이므로
    # 즉시 성공했다고 가정한다.
    # ----------------------------------------------

    if control.command == OPEN:
        equalization_active = True
        open_count += 1

    elif control.command == CLOSE:
        equalization_active = False
        close_count += 1

    # ----------------------------------------------
    # 최종 payload
    # ----------------------------------------------

    payload = {
        "timestamp": timestamp.isoformat(),
        "current_depth_m": round(
            depth_m,
            4,
        ),
        "forecast_status": (
            forecast.status
        ),
        "state": (
            risk.state
        ),
        "t_warning_min": (
            risk.t_warning_min
        ),
        "t_external_critical_min": (
            risk.t_external_critical_min
        ),
        "command": (
            control.command
        ),
        "equalization_active": (
            equalization_active
        ),
    }

    payloads.append(payload)

    print(payload)


# ==================================================
# 5. Automatic assertions
# ==================================================

# 처음 29개는 30분 입력이 아직 안 쌓였으므로 WARMUP
assert all(
    p["forecast_status"] == "WARMUP"
    for p in payloads[:29]
)

# 30번째부터는 READY
assert payloads[29]["forecast_status"] == "READY"

assert all(
    p["forecast_status"] == "READY"
    for p in payloads[29:]
)


# mock 물리 조건에서 실제 OPEN branch가
# 최소 한 번 실행돼야 함
assert open_count >= 1, (
    "OPEN command가 한 번도 발생하지 않았습니다."
)


# OPEN 후 조건이 해제되거나 대피가 완료되면
# CLOSE가 발생해야 함
assert close_count >= 1, (
    "CLOSE command가 한 번도 발생하지 않았습니다."
)


# 현재 외부수위가 실제 critical 0.30m 이상인 순간은
# LED 상태가 DANGER여야 한다.
critical_payloads = [
    p
    for p in payloads
    if p["current_depth_m"] >= 0.30
]

assert len(critical_payloads) > 0

assert all(
    p["state"] == DANGER
    for p in critical_payloads
)


# 현재 0.30m 이상이면
# external critical time은 0
assert all(
    p["t_external_critical_min"] == 0
    for p in critical_payloads
)


print()
print("==============================")
print("E2E TEST PASSED")
print("==============================")
print(f"Total samples : {len(payloads)}")
print(f"OPEN count    : {open_count}")
print(f"CLOSE count   : {close_count}")