from __future__ import annotations

from dataclasses import dataclass

from meit_model.risk import RiskResult


NONE = "NONE"
OPEN = "OPEN"
CLOSE = "CLOSE"


@dataclass
class ControllerResult:
    command: str

    # LED / 사용자 경고 상태
    state: str

    # 사람이 확인하기 위한 설명
    reason: str


class SafetyController:
    """
    압력평형 장치 제어 명령을 결정한다.

    중요:
    RiskResult.state(SAFE/WARNING/DANGER)는
    LED/음성 경고 상태이며,
    DANGER 자체가 OPEN 조건은 아니다.

    OPEN 여부는 별도의 물리 판단 결과를 사용한다.
    """

    @staticmethod
    def _require_bool(name: str, value: bool) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool, got {type(value).__name__}: {value!r}")
        return value

    # equalization_active: True = confirmed OPEN, False = confirmed CLOSED.
    # Never map OPENING/CLOSING/FAULT/UNKNOWN or a sent command to this bool.
    # Device acknowledgement, freshness, timeouts and retries belong to the
    # integration layer; this class is a stateless decision interface only.
    def decide(
        self,
        risk: RiskResult,
        *,
        evacuation_complete: bool,
        equalization_required: bool,
        equalization_allowed: bool,
        equalization_active: bool,
    ) -> ControllerResult:
        """
        Parameters
        ----------
        risk:
            risk.py에서 계산한 LED/조기경고 상태.

        evacuation_complete:
            모든 사람이 대피했는지 여부.

        equalization_required:
            실내·외 수위 및 등가수심 계산 결과,
            압력평형이 필요하다고 판단되었는지 여부.

            주의:
            risk.state == DANGER만으로 True로 만들면 안 된다.

        equalization_allowed:
            현재 물리적/시스템적으로
            유입구 개방이 허용되는 조건인지 여부.

        equalization_active:
            현재 압력평형 장치가 이미 열려 있는지 여부.
        """

        # Validate ALL inputs before any policy branch, including early returns.
        # A TypeError is not a NONE/CLOSE command. The caller must report the
        # fault and avoid generating a new command from this invalid sample.
        evacuation_complete = self._require_bool("evacuation_complete", evacuation_complete)
        equalization_required = self._require_bool("equalization_required", equalization_required)
        equalization_allowed = self._require_bool("equalization_allowed", equalization_allowed)
        equalization_active = self._require_bool("equalization_active", equalization_active)

        # --------------------------------------------------
        # 1. 이미 대피가 완료됨
        # --------------------------------------------------
        if evacuation_complete:
            if equalization_active:
                return ControllerResult(
                    command=CLOSE,
                    state=risk.state,
                    reason=(
                        "대피가 완료되어 "
                        "활성화된 압력평형 장치를 닫습니다."
                    ),
                )

            return ControllerResult(
                command=NONE,
                state=risk.state,
                reason=(
                    "대피가 완료되어 "
                    "압력평형 제어가 필요하지 않습니다."
                ),
            )

        # --------------------------------------------------
        # 2. 압력평형이 필요하지 않음
        # --------------------------------------------------
        if not equalization_required:
            if equalization_active:
                return ControllerResult(
                    command=CLOSE,
                    state=risk.state,
                    reason=(
                        "현재 압력평형이 필요하지 않아 "
                        "활성화된 장치를 닫습니다."
                    ),
                )

            return ControllerResult(
                command=NONE,
                state=risk.state,
                reason=(
                    "현재 압력평형이 필요하지 않아 "
                    "장치 상태를 유지합니다."
                ),
            )

        # --------------------------------------------------
        # 3. 압력평형은 필요하지만
        #    개방이 허용되지 않는 상태
        # --------------------------------------------------
        # TODO: Confirm immediate CLOSE on permission loss while OPEN with
        # the mechanical/electronics team before real automatic actuation.
        if not equalization_allowed:
            if equalization_active:
                return ControllerResult(
                    command=CLOSE,
                    state=risk.state,
                    reason=(
                        "압력평형이 필요하지만 "
                        "현재 개방 허용 조건을 만족하지 않아 "
                        "장치를 닫습니다."
                    ),
                )

            return ControllerResult(
                command=NONE,
                state=risk.state,
                reason=(
                    "압력평형이 필요하지만 "
                    "현재 개방 허용 조건을 만족하지 않습니다."
                ),
            )

        # --------------------------------------------------
        # 4. 사람 남아 있음
        #    + 압력평형 필요
        #    + 개방 허용
        # --------------------------------------------------
        if not equalization_active:
            return ControllerResult(
                command=OPEN,
                state=risk.state,
                reason=(
                    "대피가 완료되지 않았고 "
                    "압력평형이 필요하며 "
                    "개방 허용 조건을 만족하여 "
                    "압력평형 장치를 엽니다."
                ),
            )

        # 이미 열린 상태라면 OPEN 명령을 계속 보낼 필요 없음
        return ControllerResult(
            command=NONE,
            state=risk.state,
            reason=(
                "압력평형 장치가 이미 활성화되어 있어 "
                "현재 상태를 유지합니다."
            ),
        )