# 14단계: 제어 판단 인터페이스

SafetyController는 명령을 반환하는 순수 판단 함수다. 장치를 구동하거나 물리 조건을 직접 계산하지 않는다. LED 상태는 출력에 전달할 뿐 OPEN 조건으로 사용하지 않는다.

## 입력 계약

네 입력 모두 Python bool이어야 한다. None, 문자열, 정수 0/1 등을 임의 변환하지 않고 모든 분기 전에 TypeError로 거부한다. 통신 계층은 합의된 규격에 따라 값을 명시적으로 파싱해야 한다. `bool("false")`는 True이므로 사용하면 안 된다.

- `evacuation_complete`: 대피 완료가 확인됐는가. 미확인을 False로 대신하지 않는다.
- `equalization_required`: 최신 물리 판단에 따른 압력평형 필요 여부.
- `equalization_allowed`: 최신 물리·시스템 조건에 따른 개방 허용 여부.
- `equalization_active`: 실제 장치 상태가 확인된 OPEN이면 True, 확인된 CLOSED이면 False.

OPENING / CLOSING / FAULT / UNKNOWN은 False가 아니다. 이 상태에서는 통신 계층이 이 boolean 인터페이스를 호출하지 않고 별도 상태 처리를 해야 한다. OPEN 전송만으로 active=True로 바꾸지 않는다. 실제 위치 확인 응답, 데이터 신선도, 명령 ID, 중복 명령 방지, 타임아웃·재시도는 후속 연동 계층의 책임이다. Fake 장치에서 즉시 상태를 전환하는 것은 시뮬레이션에만 해당한다.

## 오류 처리

TypeError가 발생하면 ControllerResult는 반환되지 않는다. 호출자는 오류를 기록하고 해당 입력으로 새 제어 명령을 생성하지 않는다. 예외를 NONE/CLOSE로 바꾸거나 이전 OPEN 명령을 재사용하지 않는다. 이 원칙은 이미 움직이거나 열린 장치의 상태를 안전하게 만드는 보장이 아니다. 장치 유지·차단·타임아웃 정책은 전자·기계팀과 별도로 확정해야 한다.

## 현재 정책과 미확정 사항

미대피 + 평형 필요 + 개방 허용 + 확인된 CLOSED이면 OPEN. 이미 OPEN이고 계속 필요·허용 상태이면 NONE. 대피 완료, 필요 해제 또는 허용 해제 시 OPEN 상태라면 CLOSE를 반환한다.

특히 `active=True`에서 `allowed=False`로 바뀔 때 즉시 CLOSE하는 정책은 잠정 유지한다. TODO는 동작을 차단하는 장치가 아니므로 실제 자동 구동 전에 전자·기계 검토가 필요하다. 이번 단계의 완료 범위는 입력 검증된 판단 인터페이스이며, 최종 물리 제어 검증 완료가 아니다.

## 검증 명령

```powershell
python -m meit_model.test_controller
python -m unittest discover -s tests -v
```

실행 예제는 기존 7개 시나리오와 잘못된 입력을 검증한다. 자동 테스트는 16개 boolean 조합 × 3개 LED 상태 및 각 분기의 잘못된 입력 거부를 검사한다.
