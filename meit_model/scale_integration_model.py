"""Scale-model integration policy independent of the trained LSTM."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import math
from statistics import median


STATE_ORDER = {'SAFE': 0, 'WARNING': 1, 'DANGER': 2}


@dataclass(frozen=True)
class ScaleIntegrationResult:
    state: str
    external_state: str
    internal_state: str
    gap_state: str
    external_depth_m: float
    internal_depth_m: float
    depth_gap_m: float
    open_signal: bool
    close_signal: bool
    command: str
    equalization_active: bool
    warning_depth_m: float
    danger_depth_m: float
    open_depth_m: float
    internal_warning_depth_m: float
    internal_limit_depth_m: float
    warning_depth_gap_m: float
    open_depth_gap_m: float
    filter_sample_count: int = 1
    filter_ready: bool = True
    safety_override: bool = False
    decision_basis: str = 'CURRENT_DEPTH_GAP'
    prediction_source: str = 'none'

    @property
    def led(self) -> str:
        return {'SAFE': 'BLUE', 'WARNING': 'GREEN', 'DANGER': 'RED'}[self.state]

    def to_dict(self) -> dict:
        return {**asdict(self), 'led': self.led}


class ScaleIntegrationModel:
    """Current-level policy for the physical scale model.

    LED state and opening use only the outside-minus-inside depth gap.
    Individual depth states are diagnostics only.
    The class does not import, load, or invoke the LSTM or the production
    RiskEngine.
    """

    def __init__(
        self,
        *,
        warning_depth_m: float = 0.02,
        danger_depth_m: float = 0.03,
        open_depth_m: float = 0.03,
        internal_warning_depth_m: float = 0.025,
        internal_limit_depth_m: float = 0.03,
        warning_depth_gap_m: float = 0.02,
        open_depth_gap_m: float = 0.04,
    ):
        thresholds = (
            warning_depth_m,
            danger_depth_m,
            open_depth_m,
            internal_warning_depth_m,
            internal_limit_depth_m,
            warning_depth_gap_m,
            open_depth_gap_m,
        )
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v < 0 for v in thresholds):
            raise ValueError('Scale-model thresholds must be finite and nonnegative')
        if not warning_depth_m < danger_depth_m <= open_depth_m:
            raise ValueError('Thresholds must satisfy warning < danger <= open')
        if not internal_warning_depth_m < internal_limit_depth_m:
            raise ValueError('Internal thresholds must satisfy warning < limit')
        if not warning_depth_gap_m < open_depth_gap_m:
            raise ValueError('Gap thresholds must satisfy warning < danger/open')
        self.warning_depth_m = float(warning_depth_m)
        self.danger_depth_m = float(danger_depth_m)
        self.open_depth_m = float(open_depth_m)
        self.internal_warning_depth_m = float(internal_warning_depth_m)
        self.internal_limit_depth_m = float(internal_limit_depth_m)
        self.warning_depth_gap_m = float(warning_depth_gap_m)
        self.open_depth_gap_m = float(open_depth_gap_m)

    @staticmethod
    def _depth(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a number')
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
        return value

    @staticmethod
    def _state(depth_m: float, warning_depth_m: float, danger_depth_m: float) -> str:
        if depth_m >= danger_depth_m:
            return 'DANGER'
        if depth_m >= warning_depth_m:
            return 'WARNING'
        return 'SAFE'

    def evaluate(
        self,
        external_depth_m: float,
        internal_depth_m: float,
        *,
        equalization_active: bool = False,
    ) -> ScaleIntegrationResult:
        external = self._depth('external_depth_m', external_depth_m)
        internal = self._depth('internal_depth_m', internal_depth_m)
        if not isinstance(equalization_active, bool):
            raise ValueError('equalization_active must be bool')

        external_state = self._state(external, self.warning_depth_m, self.danger_depth_m)
        internal_state = self._state(
            internal,
            self.internal_warning_depth_m,
            self.internal_limit_depth_m,
        )
        # Millimetre precision is more than the scale sensor can resolve and
        # avoids binary-float boundary surprises at exactly 2/4 cm.
        gap = round(external - internal, 9)
        gap_state = self._state(
            max(0.0, gap),
            self.warning_depth_gap_m,
            self.open_depth_gap_m,
        )
        state = gap_state
        open_signal = gap >= self.open_depth_gap_m
        # Preserve the existing closing rule; opening always takes priority.
        close_signal = external < self.danger_depth_m
        if open_signal:
            command = 'OPEN'
        elif equalization_active and close_signal:
            command = 'CLOSE'
        else:
            command = 'NONE'

        return ScaleIntegrationResult(
            state=state,
            external_state=external_state,
            internal_state=internal_state,
            gap_state=gap_state,
            external_depth_m=external,
            internal_depth_m=internal,
            depth_gap_m=gap,
            open_signal=open_signal,
            close_signal=close_signal,
            command=command,
            equalization_active=equalization_active,
            warning_depth_m=self.warning_depth_m,
            danger_depth_m=self.danger_depth_m,
            open_depth_m=self.open_depth_m,
            internal_warning_depth_m=self.internal_warning_depth_m,
            internal_limit_depth_m=self.internal_limit_depth_m,
            warning_depth_gap_m=self.warning_depth_gap_m,
            open_depth_gap_m=self.open_depth_gap_m,
        )


class StableScaleIntegrationModel:
    """Scale policy using five-sample medians and a three-sample startup gate."""

    def __init__(self, policy=None, *, window_samples: int = 5, minimum_samples: int = 3):
        if (isinstance(window_samples, bool) or not isinstance(window_samples, int)
                or window_samples < 1 or window_samples % 2 == 0):
            raise ValueError('window_samples must be a positive odd integer')
        if (isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int)
                or not 1 <= minimum_samples <= window_samples):
            raise ValueError('minimum_samples must be within the filter window')
        self.policy = policy or ScaleIntegrationModel()
        self.minimum_samples = minimum_samples
        self.samples = deque(maxlen=window_samples)

    def update(
        self,
        external_depth_m: float,
        internal_depth_m: float,
        *,
        equalization_active: bool = False,
    ) -> ScaleIntegrationResult:
        # Validate before adding a pair so an invalid sensor value cannot
        # partially update the filter.
        external = self.policy._depth('external_depth_m', external_depth_m)
        internal = self.policy._depth('internal_depth_m', internal_depth_m)
        if not isinstance(equalization_active, bool):
            raise ValueError('equalization_active must be bool')
        self.samples.append((external, internal))
        filtered_external = median(sample[0] for sample in self.samples)
        filtered_internal = median(sample[1] for sample in self.samples)
        result = self.policy.evaluate(
            filtered_external,
            filtered_internal,
            equalization_active=equalization_active,
        )
        ready = len(self.samples) >= self.minimum_samples
        # A single splash or startup sample may change the LED diagnostic, but
        # it cannot actuate opening before enough paired readings exist.
        return replace(
            result,
            command=result.command if ready else 'NONE',
            filter_sample_count=len(self.samples),
            filter_ready=ready,
        )
