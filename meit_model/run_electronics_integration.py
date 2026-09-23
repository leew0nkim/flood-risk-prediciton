from datetime import datetime

from meit_model.electronics_io import (
    ElectronicsSerial,
)

from meit_model.predictor import (
    ForecastPredictor,
    LiveForecastBuffer,
)

from meit_model.risk import RiskEngine


# 실제 전자팀 환경에 맞춰 수정
PORT = "COM3"
BAUDRATE = 9600


CHECKPOINT = (
    "artifacts/risk_lstm_cv_v2/"
    "fold_1/best_model.pt"
)


predictor = ForecastPredictor(
    CHECKPOINT
)

live = LiveForecastBuffer(
    predictor
)

risk_engine = RiskEngine(
    warning_current_depth_m=0.25,
    external_critical_depth_m=0.30,
    forecast_warning_depth_m=0.29,
    danger_horizon_min=5,
)

electronics = ElectronicsSerial(
    port=PORT,
    baudrate=BAUDRATE,
)


print("Electronics integration started.")


try:
    while True:

        # ==========================================
        # 1. 전자시스템 → AI
        # ==========================================

        packet = (
            electronics.read_sensor_packet()
        )

        if packet is None:
            continue

        print(
            "RX:",
            packet,
        )


        # ==========================================
        # 2. 이미 meter 단위
        # ==========================================

        external_depth_m = (
            packet.external_depth_m
        )


        # ==========================================
        # 3. Predictor
        # ==========================================

        forecast = live.add_measurement(
            external_depth_m,
            timestamp=datetime.now(),
        )


        # ==========================================
        # 4. Risk
        # ==========================================

        risk = risk_engine.evaluate(
            current_depth_m=external_depth_m,
            forecast=forecast,
        )


        # ==========================================
        # 5. AI → 전자시스템
        #
        # 1차 연동에서는 실제 장치 제어 금지
        # ==========================================

        payload = {
            "state": (
                risk.state
            ),

            "forecast_status": (
                forecast.status
            ),

            "t_warning_min": (
                risk.t_warning_min
            ),

            "t_external_critical_min": (
                risk.t_external_critical_min
            ),

            "command": "NONE",
        }


        electronics.send_ai_result(
            payload
        )

        print(
            "TX:",
            payload,
        )


finally:
    electronics.close()