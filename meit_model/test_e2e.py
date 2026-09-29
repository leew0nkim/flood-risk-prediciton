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
)


CHECKPOINT = (
    "artifacts/risk_lstm_cv_v2/"
    "fold_1/best_model.pt"
)

START_TIME = datetime(
    2026, 9, 23, 9, 0
)


# ============================================================
# 1. 실제 predictor / risk / controller 준비
# ============================================================

predictor = ForecastPredictor(
    CHECKPOINT
)

live = LiveForecastBuffer(
    predictor
)

risk_engine = RiskEngine(
    warning_current_depth_m=0.25,
    external_critical_depth_m=0.30,
    # Use the canonical critical threshold until a calibration artifact for
    # the current data fingerprint is explicitly supplied.
    forecast_warning_depth_m=0.30,
    danger_horizon_min=5,

    internal_warning_depth_m=0.25,
    internal_critical_depth_m=0.30,

    pressure_critical_equivalent_depth_m=0.30,
)

controller = SafetyController()


# ============================================================
# 2. Fake Sensor 데이터
#
# 외부수위:
# - 처음 30분 0.20m
# - 이후 0.205 → 0.305m 상승
#
# 내부수위:
# - 처음에는 0.05m
# - 유입구가 열린 뒤 조금씩 증가한다고 가정
#
# NOTE:
# 실제 물리 시뮬레이션이 아니라
# software integration 확인용 fake data이다.
# ============================================================

external_depths_m = (
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

assert len(external_depths_m) == 51


# 유입구 OPEN 전의 기본 내부수위
internal_depth_m = 0.05

# Fake hardware 상태
equalization_active = False


results = []


# ============================================================
# 3. E2E simulation
# ============================================================

for index, external_depth_m in enumerate(
    external_depths_m
):
    minute = index + 1

    timestamp = (
        START_TIME
        + timedelta(minutes=index)
    )

    # --------------------------------------------------------
    # Fake Sensor:
    # equalization이 활성화된 동안
    # 내부수위가 조금씩 상승한다고 가정.
    #
    # 실제 물리 모델이 아님.
    # --------------------------------------------------------

    if equalization_active:
        internal_depth_m = min(
            internal_depth_m + 0.03,
            external_depth_m,
        )

    # --------------------------------------------------------
    # Predictor
    # --------------------------------------------------------

    forecast = live.add_measurement(
        external_depth_m,
        timestamp=timestamp,
    )

    # --------------------------------------------------------
    # Risk
    #
    # 새 구조:
    # 외부수위 + 내부수위 + equalization state 전달
    # --------------------------------------------------------

    risk = risk_engine.evaluate(
        current_depth_m=external_depth_m,
        internal_depth_m=internal_depth_m,
        equalization_active=equalization_active,
        forecast=forecast,
    )

    # --------------------------------------------------------
    # Physical/control 조건
    #
    # 아직 실제 equivalent-depth control module이 없으므로
    # E2E에서는 mock 조건 사용.
    #
    # minute 43~48:
    #     equalization 필요
    #
    # minute 49 이후:
    #     evacuation 완료
    # --------------------------------------------------------

    equalization_required = (
        43 <= minute < 49
    )

    equalization_allowed = True

    evacuation_complete = (
        minute >= 49
    )

    # --------------------------------------------------------
    # Controller
    # --------------------------------------------------------

    control = controller.decide(
        risk=risk,
        evacuation_complete=evacuation_complete,
        equalization_required=equalization_required,
        equalization_allowed=equalization_allowed,
        equalization_active=equalization_active,
    )

    # --------------------------------------------------------
    # Fake hardware feedback
    #
    # 실제 장치에서는 반드시 ACK / confirmed state를 받아야 함.
    # 여기서는 software E2E이므로 즉시 반영.
    # --------------------------------------------------------

    if control.command == "OPEN":
        equalization_active = True

    elif control.command == "CLOSE":
        equalization_active = False

    # --------------------------------------------------------
    # Output payload
    # --------------------------------------------------------

    payload = {
        "timestamp": (
            timestamp.isoformat()
        ),

        "external_depth_m": (
            round(external_depth_m, 4)
        ),

        "internal_depth_m": (
            round(internal_depth_m, 4)
        ),

        "forecast_status": (
            forecast.status
        ),

        "state": (
            risk.state
        ),

        "current_state": (
            risk.current_state
        ),

        "forecast_state": (
            risk.forecast_state
        ),

        "equivalent_depth_m": (
            None
            if risk.equivalent_depth_m is None
            else round(
                risk.equivalent_depth_m,
                4,
            )
        ),

        "pressure_direction": (
            risk.pressure_direction
        ),

        "t_external_warning_min": (
            risk.t_external_warning_min
        ),

        "t_external_critical_min": (
            risk.t_external_critical_min
        ),

        "t_critical_min": (
            risk.t_critical_min
        ),

        "pressure_forecast_status": (
            risk.pressure_forecast_status
        ),

        "command": (
            control.command
        ),

        "equalization_active": (
            equalization_active
        ),
    }

    results.append(
        payload
    )

    print(
        payload
    )


# ============================================================
# 4. Assertions
# ============================================================

# ------------------------------------------------------------
# A. 처음 29개는 WARMUP
# ------------------------------------------------------------

assert all(
    item["forecast_status"] == "WARMUP"
    for item in results[:29]
)


# ------------------------------------------------------------
# B. 30번째부터 READY
# ------------------------------------------------------------

assert (
    results[29]["forecast_status"]
    == "READY"
)


# ------------------------------------------------------------
# C. 내부수위가 RiskEngine까지 전달됨
# ------------------------------------------------------------

assert all(
    item["internal_depth_m"] is not None
    for item in results
)


# ------------------------------------------------------------
# D. equivalent depth가 계산됨
# ------------------------------------------------------------

assert all(
    item["equivalent_depth_m"]
    is not None
    for item in results
)


# ------------------------------------------------------------
# E. OPEN / CLOSE 각각 한 번 이상
# ------------------------------------------------------------

open_count = sum(
    item["command"] == "OPEN"
    for item in results
)

close_count = sum(
    item["command"] == "CLOSE"
    for item in results
)

assert open_count >= 1
assert close_count >= 1


# ------------------------------------------------------------
# F. 현재 외부수위 >= 0.30m이면
#    침수상태 자체가 DANGER
# ------------------------------------------------------------

for item in results:

    if item["external_depth_m"] >= 0.30:

        assert (
            item["state"]
            == DANGER
        )

        assert (
            item[
                "t_external_critical_min"
            ]
            == 0
        )


# ------------------------------------------------------------
# G. equalization이 활성화된 동안
#    미래 압력 추정은 비활성화
# ------------------------------------------------------------

active_results = [
    item
    for item in results
    if item["equalization_active"]
]

assert len(active_results) > 0

for item in active_results[1:]:

    assert (
        item[
            "pressure_forecast_status"
        ]
        == "UNAVAILABLE_EQUALIZATION_ACTIVE"
    )

    assert (
        item["t_critical_min"]
        is None
    )


# ------------------------------------------------------------
# H. LED DANGER가 자동 OPEN을 뜻하지 않음을 검증
# ------------------------------------------------------------

danger_without_open = any(
    item["state"] == DANGER
    and item["command"] == "NONE"
    for item in results
)

assert danger_without_open


# ============================================================
# 5. Summary
# ============================================================

print()
print("=" * 50)
print("E2E TEST PASSED")
print("=" * 50)

print(
    f"Total samples : "
    f"{len(results)}"
)

print(
    f"OPEN count    : "
    f"{open_count}"
)

print(
    f"CLOSE count   : "
    f"{close_count}"
)

print(
    "Risk chain    : "
    "external + internal "
    "→ equivalent depth "
    "→ T_critical "
    "→ LED"
)

print(
    "Control chain : "
    "risk + mock physical conditions "
    "→ OPEN/CLOSE/NONE"
)
