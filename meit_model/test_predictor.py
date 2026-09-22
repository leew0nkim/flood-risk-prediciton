from datetime import datetime, timedelta

from meit_model.predictor import ForecastPredictor, LiveForecastBuffer


CHECKPOINT_PATH = "artifacts/risk_lstm_cv_v2/fold_1/best_model.pt"

def main():
    predictor = ForecastPredictor(
        CHECKPOINT_PATH
    )

    recent_depths = [
        0.20,
        0.201,
        0.202,
        0.204,
        0.205,
        0.207,
        0.208,
        0.210,
        0.212,
        0.214,
        0.216,
        0.218,
        0.220,
        0.222,
        0.224,
        0.226,
        0.228,
        0.230,
        0.232,
        0.234,
        0.236,
        0.238,
        0.240,
        0.242,
        0.244,
        0.246,
        0.248,
        0.250,
        0.252,
        0.254,
    ]

    result = predictor.predict(
        recent_depths,
        origin_time=datetime(2026, 9, 23, 1, 0),
    )

    print("status:", result.status)

    for timestamp, depth in zip(
        result.forecast_times,
        result.predicted_depths_m,
    ):
        print(
            timestamp.strftime("%H:%M"),
            f"{depth:.4f} m",
        )


    assert result.status == "READY"
    live = LiveForecastBuffer(predictor)
    origin = datetime(2026, 9, 23, 1, 0)
    for i in range(35):
        result = live.add_measurement(.20 + i * .001,
                                      timestamp=origin + timedelta(minutes=i))
        assert result.status == ("WARMUP" if i < 29 else "READY")
        print(i + 1, result.status, result.message)
    result = live.add_measurement(None, timestamp=origin + timedelta(minutes=35))
    assert result.status == "INSUFFICIENT_DATA" and live.size == 0
    result = live.add_measurement(.24, timestamp=origin + timedelta(minutes=36))
    assert result.status == "WARMUP" and live.size == 1
    print("Live buffer reset and recovery passed")

if __name__ == "__main__":
    main()
