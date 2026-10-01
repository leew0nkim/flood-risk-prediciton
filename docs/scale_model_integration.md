# 축소모형 전자·기계 연동 정책

이 연동기는 학습된 LSTM과 운영용 `RiskEngine`을 사용하지 않는다. 아두이노에서 받은 현재
실외·실내 수위를 이용하며, 최종 경보와 개방은 `실외수위-실내수위` 차이로만 결정한다.
개별 실외·실내 상태는 전체 진단 출력에 남지만 LED나 개방 명령에는 사용하지 않는다.

## 현재 판정 기준

LED와 경보 상태는 최근 3개 실내·외 수위쌍의 중앙값으로 빠르게 계산한다. 모터의
`OPEN/CLOSE` 명령은 기존처럼 최근 5개 중앙값을 사용해 순간 물결에 의한 오작동을 막는다.
실내가 실외보다 높으면 수위차 제어값은 0으로 취급한다.

| 3개 경보 필터 수위차 | `state` | `led` |
| --- | --- | --- |
| 2cm 미만 | `SAFE` | `BLUE` |
| 2cm 이상, 4cm 미만 | `WARNING` | `GREEN` |
| 4cm 이상 | `DANGER` | `RED` |

`command=OPEN`은 별도의 5개 모터 필터 수위차가 4cm 이상일 때만 전송한다.

프로그램 시작 후 정상 수위쌍을 3개 받기 전에는 상태와 LED 값은 계산하지만 명령은 `NONE`으로
제한한다. 이후 5개 모터 필터의 수위차가 4cm 이상이면 장치 상태와 관계없이 `OPEN`을 보낸다.

현재 폐쇄 규칙은 수위차가 4cm 미만이고, 장치가 열려 있다고 아두이노가
`equalization_active=true`로 보고하며, 필터 외부수위가 3cm 미만일 때 `CLOSE`를 보내는
것이다. 그 외에는 `NONE`이다. 실내 3cm 자체는 현재 pull된 정책에서 즉시 폐쇄 조건이 아니다.

## 입출력

아두이노 입력은 newline으로 끝나는 JSON이며 단위는 m다.

```json
{"external_depth_m":0.06,"internal_depth_m":0.02,"evacuation_complete":false,"equalization_active":false}
```

실제 센서 모드의 기본 반환값은 다음 네 필드다.

```json
{"state":"DANGER","led":"RED","mock":false,"command":"OPEN"}
```

보드 없는 `--dry-run`만 `mock=true`다. `--full-wire-payload`를 사용하면 개별 상태, 필터
수위차, 임계값 등 전체 진단을 보드에도 보내므로 수신 버퍼가 충분할 때만 사용한다.

## 실행

보드 없이 경계값을 확인한다.

```bash
python -m meit_model.run_scale_integration --dry-run --external-depth-cm 1.9 --internal-depth-cm 0
python -m meit_model.run_scale_integration --dry-run --external-depth-cm 2 --internal-depth-cm 0
python -m meit_model.run_scale_integration --dry-run --external-depth-cm 4 --internal-depth-cm 0
```

실제 센서를 사용하되 명령을 강제로 `NONE`으로 막고 10건을 확인한다.

```bash
/opt/anaconda3/bin/python -m meit_model.run_scale_integration \
  --port /dev/cu.usbmodemXXXX --baudrate 9600 \
  --observe-only --max-messages 10
```

센서 기준과 보드의 JSON 수신을 확인한 후 실제 명령 모드로 실행한다.

```bash
/opt/anaconda3/bin/python -m meit_model.run_scale_integration \
  --port /dev/cu.usbmodemXXXX --baudrate 9600
```

기본 제어 임계값을 명시하려면 다음처럼 실행한다.

```bash
/opt/anaconda3/bin/python -m meit_model.run_scale_integration \
  --port /dev/cu.usbmodemXXXX --baudrate 9600 \
  --warning-gap-cm 2 --open-gap-cm 4 --danger-cm 3
```

Terminal에는 다음이 표시된다.

- `[RX OK]`: 아두이노가 보낸 원본 센서 JSON
- `[LIVE]`: 원본 수위, 3개 경보 필터 수위차, 5개 모터 필터 수위차와 준비 상태
- `[TX]`: PC가 아두이노로 보낸 JSON

아두이노가 `AI DATA:` 뒤에 빈 문자열을 출력하고 `STATE = UNKNOWN`을 표시하면 보드가 PC
응답 도착 전에 읽은 것이다. 센서 JSON 전송 후 newline까지 도착한 PC 응답을 읽도록 펌웨어를
구성해야 한다. 서보모터 구동과 실제 개방 완료 감지는 전자·기계 측 펌웨어가 담당한다.
