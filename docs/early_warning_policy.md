# 13단계: 외부수위 조기경고

`risk.py`의 SAFE / WARNING / DANGER는 LED·음성 경고 상태다. DANGER는 빨간 조기경고이며, 문 개방 불가능 판정이나 유입구 OPEN 명령이 아니다.

| 값 | 의미 |
| --- | --- |
| `t_warning_min` | 예측 외부수위가 보정 경고 기준 0.290m에 도달하는 예상 분 |
| `t_external_critical_min` | 외부수위 0.300m 도달 예상 분. 표시·기록용 |
| 향후 물리 모듈의 `t_critical_min` | 실내·외 수위 기반 등가수심 기준 도달 시간. 아직 미구현 |

이미 현재 외부수위가 0.300m 이상이면 두 시간은 0이다. 미래 도달 시간의 None은 예측 범위 내 도달 없음 또는 예측 불가를 의미하며, `forecast_status`와 함께 해석한다.

## LED 정책 (우선순위 순)

1. 현재 외부수위 ≥ 0.300m: DANGER.
2. 예측 불가 또는 잘못된 예측: WARNING. 준비/데이터 부족 상태는 `forecast_status`로 별도 표시한다.
3. `t_warning_min` ≤ 5분: DANGER.
4. `t_warning_min` 6~15분 또는 현재 외부수위 ≥ 0.250m: WARNING.
5. 나머지: SAFE.

TEST 4는 현재 0.22m, +4분 0.291m, 15분 내 0.300m 도달 없음이다. 결과는 DANGER / t_warning=4 / t_external_critical=None이다. 이는 4분 후 탈출 불가능이라는 뜻이 아니다.

RiskResult에는 OPEN/CLOSE 명령을 넣지 않는다. 전자팀 전달 계층에 command 필드가 필요하면 1차 연동에서는 NONE으로 고정한다. 향후 별도 controller가 실내·외 수위, 등가수심, 미대피 여부와 물리 판단 인터페이스를 받아 제어를 결정한다.

## 평가 의미

목표 operational metric은 실제 기준 도달 최소 5분 전에 노랑 이상 경고가 발생했는가이다. 빨강만을 대상으로 한 지표와 구분한다.

현재 event evaluator는 미래 15분 중 예측 외부수위 ≥ 경고 기준인 origin을 alarm으로 센다. 이는 **예측 기반** WARNING/DANGER 조건에 해당한다. 현재 수위 ≥ 0.250m 또는 예측 불가에 따른 WARNING은 포함하지 않는다. 또한 crossing 전 15분 범위의 최초 alarm을 찾으므로 전체 이력의 SAFE→WARNING 최초 전환과 동일하지 않다.

기존 수치는 예측 기반 이벤트 평가로 유지한다. 전체 LED 정책의 최초 경고 성능을 주장하려면 실제 시계열에서 RiskEngine 상태를 재생하는 별도 평가가 필요하다. 자동제어 성능은 이 평가로 검증되지 않는다.
