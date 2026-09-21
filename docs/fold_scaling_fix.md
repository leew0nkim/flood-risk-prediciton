# CV 정규화 및 asymmetric loss 수정

## 수정 범위

- 기존 processed 파일은 변경하지 않는다. 기존 scaler로 배열을 m 단위로 복원한 다음 fold별 scaler를 적용한다.
- fold scaler는 해당 fold의 학습 입력이 덮는 고유 실측 minute bin으로만 계산한다. 겹치는 윈도는 중복 계산하지 않으며, 결측 대체값·입력 범위 밖 정답·검증 구간은 포함하지 않는다.
- 기존 LSTM과 risk-aware LSTM 모두 같은 방식으로 정규화한다. 각 fold 시작에 seed를 초기화해 단독 fold 실행과 전체 실행을 일치시킨다.
- asymmetric loss는 유효 정답에 대해 `(absolute_error * weights).mean()`을 사용한다. 예측에 따라 변하는 가중치 합으로 나누지 않는다.
- checkpoint에 fold scaler, 정규화 버전, 데이터 해시, 학습/검증 인덱스 해시를 저장한다. 평가와 calibration은 이 scaler를 사용하며 데이터·검증 인덱스 불일치 및 이전 버전 checkpoint를 거부한다.
- checkpoint 선택 기준은 기존처럼 validation MAE다. 위험 지표 기반 선택 규칙을 임의로 추가하지 않았다.

## 검증

자동 테스트 32개 통과. 검증 구간과 입력 밖 정답을 바꿔도 scaler가 바뀌지 않는지, 중복 bin 제거, 마스크 보존, 과소예측 페널티 및 gradient를 검사했다.

두 trainer를 Fold 1 / 1 epoch로 실행하고 저장된 checkpoint로 Fold 1 전체 상세 평가 및 calibration을 확인했다. 결과는 `artifacts/fix_smoke_standard`, `artifacts/fix_smoke_risk`에 있다. 성능 비교용 학습 결과가 아니다.

## 전체 재학습 및 평가

기존 모델·결과는 보존되어 있다. 아래 명령은 새 디렉터리를 사용한다. 이미 checkpoint가 있는 출력 경로에는 덮어쓰지 않으므로 재실험 때는 별도 `--output-dir`를 지정한다.

```powershell
python -m meit_model.train_lstm_cv
python -m meit_model.train_risk_lstm_cv

python -m meit_model.evaluate_lstm_cv
python -m meit_model.calibrate_risk_warning

python -m meit_model.evaluate_lstm_cv --model-dir artifacts/risk_lstm_cv_fold_scaled --output artifacts/risk_lstm_cv_fold_scaled/evaluation.json
python -m meit_model.calibrate_risk_warning --model-dir artifacts/risk_lstm_cv_fold_scaled --output artifacts/risk_lstm_cv_fold_scaled/warning_threshold_sweep.json
```

학습 기본 출력은 각각 `artifacts/lstm_cv_fold_scaled`, `artifacts/risk_lstm_cv_fold_scaled`다. `--max-epochs 1 --fold 1 --output-dir <새 경로>`로 짧은 확인 실행, `--seed`로 반복 실험이 가능하다.

이전 성능표는 수정 전 결과로 남긴다. 다음 분석에서는 재학습된 두 모델을 같은 조건으로 비교하고, 이벤트별 경고·실패 사례를 추가로 확인한다. 최종 임계값 선정과 Test 평가는 아직 수행하지 않았다.
