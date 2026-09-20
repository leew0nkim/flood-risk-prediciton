# 1단계: 데이터·Loader 연결 확인

현재 기준 커밋은 `f02196d`이며, 기존 안내문의 `a4a61bc` 이후 두 종류의 시험 데이터가 추가되었다.

| split | 샘플 수 |
| --- | ---: |
| train | 95,597 |
| validation | 24,857 |
| test_temporal | 20,478 |
| test_risk | 9,421 |

`ForecastDataset("data/processed", "test")` 대신 `test_temporal`, `test_risk`를 각각 사용한다. 두 시험 데이터 모두 모델 선택이나 CV에 사용하지 않는다.

## 확인 명령

저장소 루트에서 현재 Python 환경을 사용한다.

```powershell
python -m pip install -e ".[torch]"
python -m meit_data --verify-only
python -m unittest discover -s tests -v
python -m meit_model.check_data_pipeline
```

2026-09-20 확인 결과: 전체 데이터 검증 통과, 기존 테스트 28개 통과, 네 split의 전체 배열 및 실제 Loader 배치 검사 통과. 설치된 의존성은 그대로 사용하고 `--no-build-isolation --no-deps`로 editable 설치를 완료했다.

입력은 float32 `(B,30,1)`, 정답은 float32 `(B,15,1)`, 마스크는 bool `(B,15,1)`이다. 전체 X가 유한하고 Y의 유한 위치와 마스크가 일치한다. train 한 배치에서만 masked MAE와 역전파를 확인하며, 시험 데이터의 성능은 계산하지 않는다. 정규화·split·학습은 수행하지 않는다. 이벤트 가중치는 기존 training sampler에서만 적용한다.

## 수정한 원인

Windows Git checkout의 LF → CRLF 변환으로 `event_audit.csv`, `segment_audit.csv`, `scaler.json`, `split_manifest.json`의 SHA-256이 달라져 `--verify-only`가 실패했다. 줄바꿈을 복원했을 때 기존 manifest 해시와 정확히 일치하는 파일만 복원했다. 수치·분할·scaler·manifest 해시를 다시 만들거나 검증을 우회하지 않았다.

`.gitattributes`에서 `data/processed/** -text`를 지정하여 생성된 데이터의 바이트를 그대로 보존한다. 새 checkout에서도 이 규칙을 적용하려면 해당 파일을 변경사항과 함께 커밋해야 한다.

## 다음 단계에서 구분할 기준

현재 저장된 전처리 설정 및 위험 메타데이터는 주의 0.30m / 임계 0.40m 기준이다. 최신 프로젝트 기준은 주의 0.25m / 탈출 곤란 0.30m이다. 저장된 crossing 라벨과 위험 holdout 선정이 최신 기준으로 생성되었다고 간주하면 안 된다. 기존 전처리 원본을 보존한 채 다음 평가 구현에서 최신 기준의 위험 라벨·사례 수를 별도로 계산하고, 고정된 시험 데이터는 모델 선택에서 제외한다.

외부수위 crossing 평가는 실내수위를 결합한 등가수심 기반 문 개방 위험 판단과 구분한다.
