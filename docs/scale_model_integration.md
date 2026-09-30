# 축소모형 전자·기계 연동 정책

이 연동기는 학습된 LSTM과 운영용 `RiskEngine`을 사용하거나 수정하지 않는다. 현재 센서의
실외·실내 수위만 받아 축소모형 상태와 전자보드 명령을 결정한다. 시간 예측 기준은 없다.

## 판정 기준

| 구분 | SAFE | WARNING | DANGER |
| --- | --- | --- | --- |
| 실외 수위 | 2cm 미만 | 2cm 이상, 3cm 미만 | 3cm 이상 |
| 실내 수위 | 2.5cm 미만 | 2.5cm 이상, 3cm 미만 | 3cm 이상 |
| 실외-실내 수위차 | 2cm 미만 | 2cm 이상, 3cm 미만 | 3cm 이상 |

전체 `state`는 실외 상태와 실내 상태 중 높은 단계다. 실내 수조 한계는 3cm, 실외 수조
한계는 8cm다. 예상 실내 수위 2cm는 SAFE이며, 실내 3cm는 즉시 DANGER다.
누수가 빠른 현재 모형에서는 실외수위와 수위차 모두 2cm 주의, 3cm 위험을 사용한다.

## 개폐 기준

축소모형 시연에서는 LSTM, 미래 도달시간과 실규모 문 기준을 사용하지 않는다. 현재 실외수위와
동시에 측정한 `실외-실내` 수위차만 사용한다.

| 장치 상태 | 조건 | 명령 |
| --- | --- | --- |
| 닫힘 | 실외 3cm 이상, 실내 3cm 미만, 수위차 3cm 이상 | OPEN |
| 닫힘 | 그 외 | NONE |
| 열림 | 실내 3cm 이상 또는 실외수위 6cm 미만 | CLOSE |
| 열림 | 그 외 | NONE |

실외 3cm·실내 0cm이면 수위차가 3cm이므로 개방된다. 실내 수위가 한계인 3cm에 도달하면
추가 유입을 막기 위해 즉시 닫는다. 실외가 6cm 아래로 내려가도 닫는다.

물을 직접 부을 때 생기는 물결과 초음파센서 순간값을 줄이기 위해 최근 5개 실내·외 수위쌍의
중앙값을 사용한다. 최소 3개의 정상 수위쌍을 받기 전에는 LED 상태만 표시하고 개방 명령은
`NONE`으로 유지한다. 단, 열린 상태에서 실내 원시 측정값이 3cm에 도달하면 중앙값을 기다리지
않고 즉시 `CLOSE`한다. 이는 미래 도달시간을 사용하는 판단이 아니라 현재 센서값의 노이즈 제거다.
서보모터 구동과 실제 개방 완료 감지는 전자·기계 측 펌웨어가 담당한다.

수위차는 2cm부터 WARNING, 3cm부터 DANGER이며 3cm에서 개방 조건을 만족한다.

## 입출력

아두이노 입력은 기존 newline JSON 계약을 그대로 사용한다. 단위는 m다.

```json
{"external_depth_m":0.07,"internal_depth_m":0.02,"evacuation_complete":false,"equalization_active":false}
```

기본 반환값은 작은 보드 버퍼용 축약 JSON이다.

```json
{"state":"DANGER","mock":true,"command":"OPEN"}
```

`mock=true`는 이 출력이 학습된 LSTM 결과가 아니라 축소모형 연동 정책의 값임을 나타낸다.
`--full-wire-payload`를 주면 전체 판정 내용을 출력하고 보드에도 전송하지만, 보드 수신 버퍼가
충분할 때만 사용한다.

## 실행

보드 없이 경계값을 확인한다.

```bash
python -m meit_model.run_scale_integration --dry-run --external-depth-cm 2 --internal-depth-cm 0
python -m meit_model.run_scale_integration --dry-run --external-depth-cm 3 --internal-depth-cm 0
```

아두이노 연결 후 실제 포트로 실행한다.

```bash
/opt/anaconda3/bin/python -m meit_model.run_scale_integration \
  --port /dev/cu.usbmodemXXXX --baudrate 9600
```

입력 한 건마다 응답 한 건을 보낸다. 종료는 `Ctrl+C`다. 연결 시험만 제한하려면
`--max-messages 10`처럼 지정한다.

## 실제 축소모형 연결 순서

센서를 실제 위치에 고정한 뒤 먼저 관찰 모드로 10건을 확인한다. 실제 수위로 상태는 계산하지만
전자부로 보내는 `command`는 강제로 `NONE`이며 `mock=false`다.

```bash
/opt/anaconda3/bin/python -m meit_model.run_scale_integration \
  --port /dev/cu.usbmodemXXXX --baudrate 9600 \
  --observe-only --max-messages 10
```

Terminal의 raw/filtered 실외수위가 0~8cm, 실내수위가 0~3cm 범위에서 실제 눈금과 일치하고,
아두이노가 반환 JSON을 정상 수신하는지 확인한다. 그 다음 실제 명령 모드로 실행한다.

```bash
/opt/anaconda3/bin/python -m meit_model.run_scale_integration \
  --port /dev/cu.usbmodemXXXX --baudrate 9600
```

실제 모드의 기본 출력은 `mock=false`다. 실내 3cm 하드 한계에서는 아두이노의
`equalization_active` 피드백이 false여도 `CLOSE`를 보낸다.

현장에서 기준을 바꿀 때는 코드 수정 없이 cm 단위 실행 옵션을 사용한다.

```bash
python -m meit_model.run_scale_integration --port /dev/cu.usbmodemXXXX \
  --warning-cm 2 --danger-cm 3 --open-cm 3 \
  --warning-gap-cm 2 --open-gap-cm 3 \
  --internal-warning-cm 2.5 --internal-limit-cm 3
```

아두이노가 `AI DATA:` 뒤에 빈 문자열을 출력하고 `STATE = UNKNOWN`을 표시하면 Python 출력
문제가 아니라 보드가 응답 도착 전에 읽은 것이다. 센서 JSON을 보낸 뒤 `Serial.available()`이
참이 될 때까지 제한시간을 두고 기다린 다음 newline까지 읽어야 한다. 초음파센서의 표면까지
거리 자체가 아니라, 보정 기준거리에서 현재 거리를 뺀 실제 수심을 m 단위로 보내야 한다.
