# 학습 모델을 연결하지 않는 전자시스템 연동 시험

두 모드 모두 체크포인트를 로딩하지 않는다. `prediction_source=mock`, `command=NONE`이다.
현재 공용 ForecastResult 모듈 때문에 PyTorch 패키지는 필요하지만 실제 학습 모델을 연결하지 않는다.

```bash
python -m pip install -e ".[electronics]"
```

## 아두이노 연결 직후: 받은 값 확인 + mock 응답

연결 전 준비에는 학습 모델이나 30분 이력이 필요하지 않다. 이 Mac의
`/opt/anaconda3/bin/python`에는 PyTorch와 pyserial이 설치되어 있다.
아두이노를 연결한 후 아래 명령으로 포트를 확인한다.

```bash
/opt/anaconda3/bin/python -m serial.tools.list_ports
```

아래 명령의 `/dev/cu.usbmodemXXXX`를 실제 포트로 교체하고 저장소 루트에서 실행한다.
같은 포트를 사용하는 Arduino IDE 시리얼 모니터·플로터는 닫는다.

```bash
/opt/anaconda3/bin/python -m meit_model.run_electronics_integration \
  --mode quick --scenario safe --reply-on-input \
  --port /dev/cu.usbmodemXXXX --baudrate 9600
```

아두이노도 `Serial.begin(9600)`으로 설정한다. 두 수위는 m 단위의 실제 측정값을 숫자로,
두 상태는 JSON boolean으로 보낸다. 매 패킷 끝에 newline을 붙인다(`Serial.println` 사용 가능).
예를 들어 외부 20cm, 실내 10cm의 입력 한 줄은 다음과 같다.

```json
{"external_depth_m":0.20,"internal_depth_m":0.10,"evacuation_complete":false,"equalization_active":false}
```

호스트는 정상 JSON 한 줄마다 보드의 작은 수신 버퍼에 맞춘 mock JSON 한 줄을 반환한다.
`safe` 시나리오는 센서값과 무관하게 SAFE로 고정한다.

```json
{"state":"SAFE","mock":true,"command":"NONE"}
```

- PC 콘솔의 `[RX OK]`에는 실제 수신한 원문을, `[TX MOCK]`에는 반환값을 표시한다.
- `[RX ERROR]`에는 잘못된 입력 원문과 파싱 오류를 표시한다. 이런 입력에는 성공 응답을 보내지 않는다.
- 입력이 없거나 한 줄이 덜 들어왔으면 기다린다. 센서 숫자만 보낸 CSV나 내부 수위 null은 현재 계약에서 유효하지 않다.
- 아두이노는 newline까지 응답을 읽는다. 기본 응답은 64바이트 미만이다. 전체 위험 진단 필드는 Terminal 출력에만 남긴다.
- 처음에는 정상 패킷 하나를 보내고 응답을 읽은 뒤, 약 1초 간격으로 다음 패킷을 보낸다.
  응답 timeout이 나면 재시도하거나 오류를 표시한다. 통신 채널에 일반 디버그 문자열을 센서 패킷과 섞지 않는다.
- LED 시험 시 `--scenario warning` 또는 `--scenario danger`로 바꿀 수 있다. 응답의 `command`는 항상 NONE이다.
- `--max-messages 3`을 추가하면 정상 요청에 3번 응답한 뒤 종료한다. 그렇지 않으면 Ctrl+C로 종료한다.

`--reply-on-input` 없이 실행하는 기존 quick은 입력과 무관하게 주기적으로 전체 시험 결과를 송신한다.
아두이노 보드 자체의 센서 읽기·LED 제어 코드는 해당 펌웨어에서 구현해야 한다.

## quick: 통신·경고 시나리오 시험

```bash
python -m meit_model.run_electronics_integration --mode quick --port COM3
```

기본 속도는 **9600 baud**다. 실제 포트로 교체한다(macOS 예: `/dev/cu.usbmodem...`).
센서 수신이나 30분 이력 없이 가상 실내·외 수위와 가상 외부 예측을 만든다.
기본 `cycle`은 5초마다 safe → warning → danger 시나리오를 반복한다.
실수신값은 `received_sensor_depth_m`, `received_internal_depth_m`에 별도로 기록하며 시나리오 판정에 쓰지 않는다.
quick은 실제 위험 감시 용도가 아니다.

| `--scenario` | 현재 외부 / 실내 | mock 외부 예측 | 결과 |
| --- | --- | --- | --- |
| safe | 0.20 / 0.10m | 모두 0.20m | SAFE |
| warning | 0.20 / 0.10m | +10분부터 0.50m | WARNING, 조건부 압력 도달 +10분 |
| danger | 0.20 / 0.10m | +3분부터 0.50m | DANGER, 조건부 압력 도달 +3분 |
| equalized | 0.50 / 0.50m | 모두 0.50m | 등가수심 0, 침수 경고 DANGER |
| inside-high | 0.20 / 0.35m | 모두 0.20m | 실내 침수 DANGER, 내부 우세 압력 |

`--scenario-seconds`는 cycle의 순환 간격만 바꾼다. 실제 시간이나 센서 이력을 가속하지 않는다.
보드 없이 출력 확인:

```bash
python -m meit_model.run_electronics_integration --mode quick --dry-run --scenario equalized --max-messages 1
```

## accumulate: 실제 센서 입력·30분 이력 시험

```bash
python -m meit_model.run_electronics_integration --mode accumulate --port COM3
```

보드는 동일 시점의 실내·외 수위와 두 boolean 상태를 JSON 한 줄로 송신한다.
두 수위 모두 같은 문 하단을 기준으로 영점을 보정한 m 단위 숫자여야 한다. **내부 수위도 필수이며 null을 허용하지 않는다.**

```json
{"external_depth_m":0.20,"internal_depth_m":0.10,"evacuation_complete":false,"equalization_active":false}
```

- 실제 외부 수위만 모델 입력 형식의 30분 이력에 누적한다. 모델 입력 차원은 바꾸지 않았다.
- 실내 수위는 같은 정상 패킷의 최신값으로 현재 침수·차압 계산에 즉시 사용한다. 30분 준비 전에도 실내 DANGER를 표시한다.
- 두 수위를 따로 갱신하거나 새 외부값에 이전 실내값을 조합하지 않는다. 잘못된 패킷은 전체 거부한다.
- 30분 구간 확보 전 `WARMUP`; 품질 기준을 통과하면 `READY`.
- READY 후 마지막 완료 bin의 외부 수심을 15번 반복하는 **persistence mock**을 만든다. 상승을 학습·예측하는 LSTM 출력이 아니다.
- 실내 관측값을 미래에도 유지한다고 가정한 조건부 압력 시간은 별도 필드로 표시한다. 평형 장치가 작동 중이면 제공하지 않는다.
- 두 모드 모두 자동 개폐 명령을 보내지 않는다. 대피 여부는 수신·검증하지만 이번 시험에서 제어하지 않는다.

현재 프로토콜에는 센서 측정 시각이 없어 호스트의 완전한 패킷 수신 시각을 사용한다.
과거 패킷을 한꺼번에 보내는 것으로 30분 관측을 대신할 수 없다. 분당 여러 번 송신하는 것을 권장한다.
고정된 `(T-60초,T]` 구간의 마지막 관측을 대표값으로 사용한다.
실제 관측 bin 최소 27개, 연속 결측 최대 2개, 마지막 관측 경과 최대 180초를 요구하며 허용 범위에서 과거값만으로 채운다.
`history_bins`는 확보한 시간 구간 수(최대 30), `observed_history_bins`는 실제 관측이 있는 구간 수다.
30개라는 숫자만 보지 말고 `forecast_status`를 함께 본다.

마지막 정상 수신 후 180초를 초과하면 STALE / INSUFFICIENT_DATA가 되고 예측·차압·도달시간은 null이다.
현재 수위 필드에는 마지막 수신값이 남으므로 `sensor_status`를 함께 해석한다.
최종 경고는 최소 WARNING이고, 이전에 확인한 DANGER는 새 정상 관측이 들어올 때까지 유지한다.
180초보다 긴 공백 후 정상 패킷이 들어오면 이력을 초기화하고 다시 누적한다.
센서 없이 기다리기만 해서는 READY가 되지 않는다.

## 출력 해석

전체 물리식·가정·임계값은 [경고 정책](early_warning_policy.md)에 정리했다.

| 필드 | 의미 |
| --- | --- |
| `mode`, `prediction_source` | quick/accumulate, 항상 mock |
| `current_depth_m`, `internal_depth_m` | 판정에 사용한 외부·내부 수위; quick은 가상값 |
| `received_sensor_depth_m`, `received_internal_depth_m` | 실제 마지막 수신값 |
| `sensor_status`, `sensor_age_seconds` | NO_DATA/OK/STALE 및 수신 후 경과 |
| `history_status`, `history_bins`, `observed_history_bins` | 입력 이력 준비 상태; quick은 BYPASSED |
| `forecast_status`, `predicted_depths_m` | 외부 예측 상태와 +1~+15분 mock 배열 |
| `forecast_origin_unix_seconds` | 예측 기준 시각; accumulate는 마지막 완료 bin 끝 |
| `state` | 전체 LED 경고; 세 위험 중 가장 높은 단계 |
| `external_state`, `internal_state`, `pressure_state` | 외부 침수 / 실내 침수 / 현재 수압 차이 경고 |
| `depth_gap_m` | 외부−실내 수위, 부호 있음 |
| `equivalent_depth_m`, `pressure_direction` | 등가수심 크기와 압력 우세 방향 |
| `bottom_pressure_difference_pa` | 공통 문 하단에서의 압력 차, 문 전체 힘 아님 |
| `pressure_status` | 부분 침수 가정의 계산 또는 자료 부족 상태 |
| `pressure_warning_equivalent_depth_m`, `pressure_critical_equivalent_depth_m` | 등가수심 잠정 경고 / 기획서 임계값 |
| `t_warning_min`, `t_external_critical_min` | 기존 외부 0.29 / 0.30m 도달시간 |
| `pressure_forecast_status` | CONDITIONAL_INTERNAL_CONSTANT 또는 UNAVAILABLE 계열 |
| `t_pressure_critical_if_internal_constant_min` | 실내 수위 고정 조건에서 외부 우세 등가수심 0.40m 도달시간 |
| `serial_input_error` | 가장 최근 파싱 오류; 다음 정상 패킷에서 해제 |
| `command` | 항상 NONE |

아두이노가 같은 포트로 보내는 진단 문자열도 구조화한다. 이 메시지들은 더 이상 JSON 파싱 오류가 아니다.

| 아두이노 원문 | `serial_input_event` | 누적 상태 필드 |
| --- | --- | --- |
| `Arduino Ready` | `BOARD_READY` | `board_ready=true` |
| `SENSOR ERROR` | `SENSOR_ERROR` | `board_sensor_status=ERROR` |
| 정상 센서 JSON | `SENSOR_DATA` | `board_sensor_status=OK` |
| `AI DATA: ...` | `AI_ECHO` | `board_echo_status=COMPLETE/TRUNCATED` |
| `STATE = SAFE/WARNING/DANGER/UNKNOWN` | `BOARD_STATE` | `board_reported_state` |
| `LED = GREEN/YELLOW/RED/OFF/UNKNOWN` | `BOARD_LED` | `board_reported_led` |

`serial_input_error`는 필수 센서 필드가 없는 JSON, 손상된 JSON, 알 수 없는 문자열에만 설정된다.
`serial_input_event`는 polling timeout이 발생해도 가장 최근에 실제로 받은 이벤트를 유지한다.
`board_sensor_status=ERROR`는 파싱 실패가 아니라 아두이노가 보고한 실제 센서 오류다.
`board_echo_status=TRUNCATED`는 아두이노가 AI 응답 전체를 보관하지 못했다는 뜻이며 LED 수신 버퍼 크기를 확인해야 한다.

None인 도달시간은 도달 없음 또는 계산 불가다. 예측/수압 상태 필드를 함께 확인한다.
압력 도달시간은 실제 탈출 잔여시간이나 문 개방 불가 판정이 아니다.

보드로 보내는 기본 전송은 newline으로 끝나는 64바이트 미만 JSON이다. 전체 진단 JSON은 Terminal에 출력한다.
보드 펌웨어가 큰 수신 버퍼와 전체 JSON 파싱을 지원할 때만 `--full-wire-payload`를 명시한다.
프로그램은 송신 후 `max(1초, 패킷 바이트 수×10/baud×1.1)` 간격을 둔다.
입력 4096바이트 초과 줄은 폐기하고, timeout으로 나뉜 입력은 재조립한다.
NaN/무한대/음수/bool/문자열 수위와 잘못된 상태값은 거부한다.
OS 연결 오류 시 포트를 닫고 종료한다. 자동 재연결이나 장치 watchdog은 구현하지 않았다.

## 검증

```bash
python -m unittest discover -s tests -v
```

가상 시계로 30분 이력·지연·결측·회복을 검증하고, 같은 gap의 서로 다른 등가수심,
역방향 압력, 수압 평형 중 침수 경고, 내부 누락, 조건부 미래 시간, STALE DANGER 유지,
분할 JSON과 임시 시리얼 송수신을 검사한다. 실제 accumulate 실행의 시간을 가속하지 않는다.
실제 센서 영점 보정, 보드 수신 버퍼·LED 반응, 문 힘 측정은 장비 실험으로 확인해야 한다.
