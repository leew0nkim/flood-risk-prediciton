"""Flood warnings and a provisional hydrostatic pressure indicator.

Metre depths must share the door-bottom datum. The equivalent-depth model
assumes still water of equal density and both surfaces below the door top.
It does not establish door operability and never issues actuator commands.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional, Sequence

from meit_model.predictor import ForecastResult

SAFE = 'SAFE'
WARNING = 'WARNING'
DANGER = 'DANGER'
_SEVERITY = {SAFE: 0, WARNING: 1, DANGER: 2}


@dataclass
class RiskResult:
    # Preserve the existing positional constructor contract used by controller.
    state: str
    current_depth_m: float  # External depth, retained for compatibility.
    warning_depth_m: float  # External forecast threshold, NOT equivalent depth.
    external_critical_depth_m: float
    t_warning_min: Optional[int]
    t_external_critical_min: Optional[int]
    forecast_status: str
    reason: str
    internal_depth_m: Optional[float] = None
    external_state: str = WARNING
    internal_state: str = WARNING
    current_state: str = WARNING
    forecast_state: str = WARNING
    t_critical_min: Optional[int] = None
    depth_gap_m: Optional[float] = None
    equivalent_depth_m: Optional[float] = None
    bottom_pressure_difference_pa: Optional[float] = None
    pressure_direction: Optional[str] = None
    pressure_status: str = 'MISSING_INTERNAL_DEPTH'
    pressure_critical_equivalent_depth_m: float = .30
    pressure_forecast_status: str = 'UNAVAILABLE'
    # Outside-higher crossing, measured from the EXTERNAL forecast origin.
    # Compatibility alias for t_critical_min; conditional estimate only.
    t_pressure_critical_if_internal_constant_min: Optional[int] = None

    @property
    def t_external_warning_min(self) -> Optional[int]:
        """External forecast-threshold diagnostic; not an LED rule."""
        return self.t_warning_min

    @property
    def current_external_depth_m(self) -> float:
        return self.current_depth_m

    @property
    def current_internal_depth_m(self) -> Optional[float]:
        return self.internal_depth_m

    @property
    def current_equivalent_depth_m(self) -> Optional[float]:
        return self.equivalent_depth_m


class RiskEngine:
    """LED = max(current indoor/outdoor flood, conditional future pressure).

    T_critical uses the current indoor depth held constant and outside-higher
    equivalent depth >= .30 m. It is a model-based estimate, not a measured
    door-operability guarantee. Unknown/active equalization disables future
    extrapolation, but does not erase an already reached current threshold.
    """

    def __init__(self, *, warning_current_depth_m: float = .25,
                 external_critical_depth_m: float = .30,
                 forecast_warning_depth_m: float = .30,
                 danger_horizon_min: int = 5,
                 internal_warning_depth_m: float = .25,
                 internal_critical_depth_m: float = .30,
                 pressure_critical_equivalent_depth_m: float = .30):
        thresholds = {
            'warning_current_depth_m': warning_current_depth_m,
            'external_critical_depth_m': external_critical_depth_m,
            'forecast_warning_depth_m': forecast_warning_depth_m,
            'internal_warning_depth_m': internal_warning_depth_m,
            'internal_critical_depth_m': internal_critical_depth_m,
            'pressure_critical_equivalent_depth_m': pressure_critical_equivalent_depth_m,
        }
        for name, value in thresholds.items():
            setattr(self, name, self._validate_current_depth(value))
        for low, high in [(warning_current_depth_m, external_critical_depth_m),
                          (internal_warning_depth_m, internal_critical_depth_m)]:
            if not low < high:
                raise ValueError('Warning depth must be below critical depth')
        if self.pressure_critical_equivalent_depth_m <= 0:
            raise ValueError('Equivalent critical depth must be positive')
        if not 0 < forecast_warning_depth_m <= external_critical_depth_m:
            raise ValueError('Invalid external forecast warning threshold')
        if (isinstance(danger_horizon_min, bool) or not isinstance(danger_horizon_min, Real)
                or not math.isfinite(danger_horizon_min)
                or int(danger_horizon_min) != danger_horizon_min
                or not 1 <= danger_horizon_min <= 15):
            raise ValueError('danger_horizon_min must be an integer in 1..15')
        self.danger_horizon_min = int(danger_horizon_min)

    @staticmethod
    def _validate_current_depth(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError('Depth must be a number in metres')
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError('Depth is out of range') from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError('Depth must be finite and nonnegative')
        return value

    @staticmethod
    def _first_crossing_min(predictions_m: Sequence[float], threshold_m: float) -> Optional[int]:
        return next((i for i, value in enumerate(predictions_m, 1) if value >= threshold_m), None)

    @staticmethod
    def _depth_state(depth, warning, critical, *, roundoff_tolerance=0.):
        def reached(threshold):
            return depth >= threshold or math.isclose(depth, threshold, rel_tol=roundoff_tolerance)
        return DANGER if reached(critical) else WARNING if reached(warning) else SAFE

    @staticmethod
    def _pressure_reached(depth, threshold):
        # sqrt/subtraction can place an exact analytical boundary just below it.
        return depth >= threshold or math.isclose(depth, threshold, rel_tol=1e-12)

    @staticmethod
    def _equivalent_depth(outside: float, inside: float) -> float:
        # sqrt(abs(outside**2-inside**2)), without overflowing from squaring.
        high, low = max(outside, inside), min(outside, inside)
        ratio = 0. if high == low else ((high - low) / high) * (1 + low / high)
        return high * math.sqrt(min(1., ratio))

    def evaluate(self, current_depth_m: float, forecast: ForecastResult, *,
                 internal_depth_m: Optional[float] = None,
                 equalization_active: Optional[bool] = None) -> RiskResult:
        outside = self._validate_current_depth(current_depth_m)
        inside = None if internal_depth_m is None else self._validate_current_depth(internal_depth_m)
        if equalization_active is not None and not isinstance(equalization_active, bool):
            raise ValueError('equalization_active must be bool or unknown (None)')
        status, predictions = forecast.status, None
        if status == 'READY':
            try:
                if forecast.predicted_depths_m is None or len(forecast.predicted_depths_m) != 15:
                    raise ValueError('Expected 15 future external depths')
                predictions = [self._validate_current_depth(v) for v in forecast.predicted_depths_m]
            except (ValueError, TypeError, OverflowError):
                status = 'INVALID_FORECAST'
        tw = self._first_crossing_min(predictions, self.forecast_warning_depth_m) if predictions is not None else None
        te = self._first_crossing_min(predictions, self.external_critical_depth_m) if predictions is not None else None
        external_state = self._depth_state(outside, self.warning_current_depth_m, self.external_critical_depth_m)
        if outside >= self.forecast_warning_depth_m:
            tw = 0
        if outside >= self.external_critical_depth_m:
            te = 0

        result = RiskResult(
            state=external_state, current_depth_m=outside,
            warning_depth_m=self.forecast_warning_depth_m,
            external_critical_depth_m=self.external_critical_depth_m,
            t_warning_min=tw, t_external_critical_min=te, forecast_status=status,
            reason='', internal_depth_m=inside, external_state=external_state,
            pressure_critical_equivalent_depth_m=self.pressure_critical_equivalent_depth_m,
        )
        if inside is not None:
            result.internal_state = self._depth_state(inside, self.internal_warning_depth_m, self.internal_critical_depth_m)
            result.depth_gap_m = outside - inside
            result.equivalent_depth_m = self._equivalent_depth(outside, inside)
            result.pressure_direction = ('OUTSIDE_HIGHER' if outside > inside else
                                         'INSIDE_HIGHER' if inside > outside else 'BALANCED')
            # Signed gauge pressure difference at the common door-bottom datum.
            bottom_pressure = 1000. * 9.80665 * result.depth_gap_m
            result.bottom_pressure_difference_pa = bottom_pressure if math.isfinite(bottom_pressure) else None
            result.pressure_status = 'ASSUMED_PARTIAL_SUBMERGENCE'
            if not math.isfinite(bottom_pressure):
                result.pressure_status = 'NUMERIC_OUT_OF_RANGE'
                result.pressure_forecast_status = 'UNAVAILABLE_NUMERIC_RANGE'
            elif outside > inside and self._pressure_reached(
                    result.equivalent_depth_m, self.pressure_critical_equivalent_depth_m):
                # Current paired observations do not require a forecast or an
                # assumption that indoor depth will remain constant.
                result.pressure_forecast_status = 'CURRENT_CRITICAL_REACHED'
                result.t_pressure_critical_if_internal_constant_min = 0
            elif equalization_active is not False:
                result.pressure_forecast_status = ('UNAVAILABLE_EQUALIZATION_ACTIVE' if equalization_active
                                                   else 'UNAVAILABLE_EQUALIZATION_STATE')
            elif predictions is not None:
                result.pressure_forecast_status = 'CONDITIONAL_INTERNAL_CONSTANT'
                result.t_pressure_critical_if_internal_constant_min = next((
                    minute for minute, depth in enumerate(predictions, 1)
                    if depth > inside and self._pressure_reached(self._equivalent_depth(depth, inside), self.pressure_critical_equivalent_depth_m)
                ), None)
        result.t_critical_min = result.t_pressure_critical_if_internal_constant_min
        result.current_state = max((result.external_state, result.internal_state), key=_SEVERITY.get)
        if result.t_critical_min == 0:
            result.forecast_state = DANGER
        elif result.pressure_forecast_status != 'CONDITIONAL_INTERNAL_CONSTANT':
            result.forecast_state = WARNING
        elif result.t_critical_min is None:
            result.forecast_state = SAFE
        elif result.t_critical_min <= self.danger_horizon_min:
            result.forecast_state = DANGER
        else:
            result.forecast_state = WARNING
        result.state = max((result.current_state, result.forecast_state), key=_SEVERITY.get)
        result.reason = (f'Current flood={result.current_state}; future pressure={result.forecast_state}; '
                         f'T_critical={result.t_critical_min} min ({result.pressure_forecast_status}). '
                         'Indoor depth held constant; no actuator command or verified door deadline.')
        return result
