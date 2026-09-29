# 1단계: 데이터·Loader 연결 확인

현재 canonical 데이터는 chronology-stratified SHA-256 규칙으로 고정한 위험 날짜를 사용한다.

| split | 샘플 수 |
| --- | ---: |
| train | 97,056 |
| validation | 24,857 |
| test_temporal | 20,478 |
| test_risk | 7,962 |

`ForecastDataset("data/processed", "test")` 대신 `test_temporal`, `test_risk`를 각각 사용한다. 두 시험 데이터 모두 모델 선택이나 CV에 사용하지 않는다.

## 확인 명령

저장소 루트에서 현재 Python 환경을 사용한다.

```powershell
python -m pip install -e ".[torch]"
python -m meit_data --verify-only
python -m unittest discover -s tests -v
python -m meit_model.check_data_pipeline
```

2026-09-29 확인 결과: 전체 데이터 검증 통과, 테스트 92개 통과, 네 split의 전체 배열 및 실제 Loader 배치 검사 통과.

입력은 float32 `(B,30,1)`, 정답은 float32 `(B,15,1)`, 마스크는 bool `(B,15,1)`이다. 전체 X가 유한하고 Y의 유한 위치와 마스크가 일치한다. train 한 배치에서만 masked MAE와 역전파를 확인하며, 시험 데이터의 성능은 계산하지 않는다. 정규화·split·학습은 수행하지 않는다. 이벤트 가중치는 기존 training sampler에서만 적용한다.

## 수정한 원인

Windows Git checkout의 LF → CRLF 변환으로 `event_audit.csv`, `segment_audit.csv`, `scaler.json`, `split_manifest.json`의 SHA-256이 달라져 `--verify-only`가 실패했다. 줄바꿈을 복원했을 때 기존 manifest 해시와 정확히 일치하는 파일만 복원했다. 수치·분할·scaler·manifest 해시를 다시 만들거나 검증을 우회하지 않았다.

`.gitattributes`에서 `data/processed/** -text`를 지정하여 생성된 데이터의 바이트를 그대로 보존한다. 새 checkout에서도 이 규칙을 적용하려면 해당 파일을 변경사항과 함께 커밋해야 한다.

## 다음 단계에서 구분할 기준

현재 저장된 전처리 설정과 위험 메타데이터는 외부수위 주의 0.25m / critical crossing 0.30m 기준으로 다시 생성되었다. `test_risk`는 0.30m를 아래에서 도달하는 평가 가능 이벤트 9개를 포함하며, 선택 날짜는 2023-09-29, 2024-10-18, 2025-08-21, 2026-04-19 UTC로 고정한다. 날짜는 시간순 후보를 네 구간으로 나눈 뒤 각 구간에서 seed 42의 SHA-256 순서가 가장 앞선 날짜를 고르는 규칙으로 선택하며 모델 예측·오차를 입력으로 사용하지 않는다.

외부수위 crossing critical 0.30m와 압력 등가수심 critical 0.30m는 숫자만 같고 평가량이 다르다. 전자는 LSTM이 출력한 외부수위 예측 평가에, 후자는 실내·외 수위를 결합한 RiskEngine 압력 정책에 사용한다. 두 결과를 별도 필드와 보고서로 기록한다.

기존 CV·calibration·checkpoint는 이전 데이터 fingerprint와 코드 커밋을 붙여 보존한다. 새 canonical 데이터의 근거로 재사용하지 않으며, 새 CV 분할과 seed 42·43·44 학습, window calibration, 5분 event 평가 및 정책 고정이 끝날 때까지 `test_temporal`과 `test_risk`에서 모델 성능을 계산하지 않는다. 구조·해시 검증은 점수 계산 없이 수행한다.

외부수위 crossing 평가는 실내수위를 결합한 등가수심 기반 문 개방 위험 판단과 구분한다.
