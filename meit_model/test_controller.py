from meit_model.controller import (
    SafetyController,
    NONE,
    OPEN,
    CLOSE,
)

from meit_model.risk import RiskResult


controller = SafetyController()


def make_risk(state):
    return RiskResult(
        state=state,
        current_depth_m=0.20,
        warning_depth_m=0.29,
        external_critical_depth_m=0.30,
        t_warning_min=None,
        t_external_critical_min=None,
        forecast_status="READY",
        reason="test",
    )


def main():
    # --------------------------------------------------
    # TEST 1
    # DANGER지만 물리적으로 압력평형 불필요
    # → NONE
    # --------------------------------------------------

    result = controller.decide(
        make_risk("DANGER"),
        evacuation_complete=False,
        equalization_required=False,
        equalization_allowed=True,
        equalization_active=False,
    )

    assert result.command == NONE
    print("\nTEST 1")
    print(result)


    # --------------------------------------------------
    # TEST 2
    # DANGER + 압력평형 필요 + 개방 허용
    # → OPEN
    # --------------------------------------------------

    result = controller.decide(
        make_risk("DANGER"),
        evacuation_complete=False,
        equalization_required=True,
        equalization_allowed=True,
        equalization_active=False,
    )

    assert result.command == OPEN
    print("\nTEST 2")
    print(result)


    # --------------------------------------------------
    # TEST 3
    # WARNING이어도 물리 조건상 압력평형 필요
    # → OPEN
    #
    # 이것은 LED 상태와 제어 판단이
    # 분리되어 있음을 검증하는 테스트.
    # --------------------------------------------------

    result = controller.decide(
        make_risk("WARNING"),
        evacuation_complete=False,
        equalization_required=True,
        equalization_allowed=True,
        equalization_active=False,
    )

    assert result.command == OPEN
    print("\nTEST 3")
    print(result)


    # --------------------------------------------------
    # TEST 4
    # 필요하지만 개방 허용 안 됨
    # → NONE
    # --------------------------------------------------

    result = controller.decide(
        make_risk("DANGER"),
        evacuation_complete=False,
        equalization_required=True,
        equalization_allowed=False,
        equalization_active=False,
    )

    assert result.command == NONE
    print("\nTEST 4")
    print(result)


    # --------------------------------------------------
    # TEST 5
    # 장치가 이미 OPEN
    # + 더 이상 압력평형 필요 없음
    # → CLOSE
    # --------------------------------------------------

    result = controller.decide(
        make_risk("SAFE"),
        evacuation_complete=False,
        equalization_required=False,
        equalization_allowed=True,
        equalization_active=True,
    )

    assert result.command == CLOSE
    print("\nTEST 5")
    print(result)


    # --------------------------------------------------
    # TEST 6
    # 대피 완료 + 장치 OPEN
    # → CLOSE
    # --------------------------------------------------

    result = controller.decide(
        make_risk("DANGER"),
        evacuation_complete=True,
        equalization_required=True,
        equalization_allowed=True,
        equalization_active=True,
    )

    assert result.command == CLOSE
    print("\nTEST 6")
    print(result)


    # --------------------------------------------------
    # TEST 7
    # 이미 필요한 상태에서 장치도 OPEN
    # → NONE (현재 상태 유지)
    # --------------------------------------------------

    result = controller.decide(
        make_risk("DANGER"),
        evacuation_complete=False,
        equalization_required=True,
        equalization_allowed=True,
        equalization_active=True,
    )

    assert result.command == NONE
    print("\nTEST 7")
    print(result)

    fields = dict(evacuation_complete=False, equalization_required=True,
                  equalization_allowed=True, equalization_active=False)
    for field, invalid in [('evacuation_complete', None),
                           ('equalization_required', 1),
                           ('equalization_allowed', 'false'),
                           ('equalization_active', None)]:
        try:
            controller.decide(make_risk('DANGER'), **{**fields, field: invalid})
        except TypeError:
            pass
        else:
            raise AssertionError(f'{field}: invalid value was accepted')
    print('Valid controller tests passed.')
    print('Invalid-input controller tests passed.')


if __name__ == '__main__':
    main()
