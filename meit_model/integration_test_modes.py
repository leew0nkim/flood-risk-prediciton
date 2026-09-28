"""Test sessions with an injected clock; no checkpoint or device access."""
from collections import deque
from datetime import datetime, timezone
import math

import numpy as np

from meit_data.inference_input import prepare_input
from meit_data.preprocess import validate_config
from meit_model.predictor import ForecastResult
from meit_model.risk import RiskEngine


class IntegrationSession:
    def __init__(self, mode, config, started_at, scenario='cycle', scenario_seconds=5):
        if mode not in ('quick', 'accumulate') or scenario not in ('cycle', 'safe', 'warning', 'danger', 'equalized', 'inside-high'):
            raise ValueError('Invalid integration test mode/scenario')
        if not math.isfinite(started_at) or not math.isfinite(scenario_seconds) or scenario_seconds <= 0:
            raise ValueError('Invalid session time/scenario duration')
        validate_config(config)
        self.mode, self.config, self.scenario = mode, config, scenario
        self.started_at, self.scenario_seconds = started_at, scenario_seconds
        # Fixed integer-second grid: exclude the partial first second.
        self.grid_origin = math.ceil(started_at)
        self.samples = deque()
        self.last_received = None
        self.last_depth = None
        self.last_internal_depth = None
        self.last_equalization_active = None
        self.last_valid_state = 'WARNING'
        self.last_now = started_at
        self.last_bin = None
        self.input_result = None
        self.history_bins = 0
        self.observed_bins = 0
        self.risk = RiskEngine()

    def update(self, packet, now):
        if not math.isfinite(now) or now < self.last_now:
            raise ValueError('Session clock must be finite and nondecreasing')
        self.last_now = now
        if packet is not None:
            depth = packet.external_depth_m
            internal = packet.internal_depth_m
            for value in (depth, internal):
                # Input preparation uses float32; do not let a finite JSON
                # number overflow into an infinite model input.
                if (isinstance(value, bool) or not isinstance(value, (float, int))
                        or not math.isfinite(value) or not 0 <= value <= np.finfo(np.float32).max):
                    raise ValueError('Invalid paired sensor depth')
            if not isinstance(packet.equalization_active, bool) or not isinstance(packet.evacuation_complete, bool):
                raise ValueError('Invalid sensor state')
            if self.last_received is not None and now - self.last_received > self.config['quality']['split_raw_gap_gt_seconds']:
                self.samples.clear()
                self.input_result = None
                self.history_bins = self.observed_bins = 0
            if self.samples and self.samples[-1][0] == now:
                self.samples.pop()
            self.samples.append((now, float(depth)))
            self.last_received, self.last_depth = now, float(depth)
            self.last_internal_depth = float(internal)
            self.last_equalization_active = packet.equalization_active
        # Keep 32 minutes plus one preceding observation for boundary/gap checks.
        while len(self.samples) > 1 and self.samples[1][0] < now - 32 * 60:
            self.samples.popleft()
        age = None if self.last_received is None else now - self.last_received
        sensor_status = 'NO_DATA' if age is None else (
            'STALE' if age > self.config['quality']['max_last_observation_age_seconds'] else 'OK')
        end = self.grid_origin + max(0, math.floor((now - self.grid_origin) / 60)) * 60
        selected_scenario = None
        if self.mode == 'quick':
            selected_scenario = self.scenario
            if selected_scenario == 'cycle':
                selected_scenario = ('safe', 'warning', 'danger')[int((now - self.started_at) / self.scenario_seconds) % 3]
            current, internal, values = {
                'safe': (.2, .1, [.2] * 15),
                'warning': (.2, .1, [.2] * 9 + [.5] * 6),
                'danger': (.2, .1, [.2] * 2 + [.5] * 13),
                'equalized': (.5, .5, [.5] * 15),
                'inside-high': (.2, .35, [.2] * 15),
            }[selected_scenario]
            forecast = ForecastResult('READY', values, None)
            equalization_active = False  # Entire quick scenario is synthetic.
            forecast_origin = now
        else:
            if now >= self.grid_origin and end != self.last_bin:
                self.last_bin = end
                available = [(t, v) for t, v in self.samples if t <= end]
                times = np.asarray([p[0] for p in available])
                values = np.asarray([p[1] for p in available])
                # Identity transform retains metres for the mock (no training scaler).
                self.input_result = prepare_input(
                    times, values, end, self.grid_origin, self.config,
                    {'depth_unit': 'm', 'mean_m': 0., 'std_m': 1.})
                self.history_bins = min(30, max(0, int((end - times[0]) // 60))) if len(times) else 0
                window_start = end - self.history_bins * 60
                recent = times[(times > window_start) & (times <= end)]
                self.observed_bins = len(np.unique(np.ceil((recent - self.grid_origin) / 60)))
            result = self.input_result
            status = result['status'] if result else 'WARMUP'
            if sensor_status == 'STALE':
                status = 'INSUFFICIENT_DATA'
            ready = status == 'READY'
            values = [float(result['X'][0, -1, 0])] * 15 if ready else None
            forecast = ForecastResult(status, values, None)
            current = self.last_depth
            internal = self.last_internal_depth
            equalization_active = self.last_equalization_active
            forecast_origin = end if ready else None
        evaluated = self.risk.evaluate(current, forecast, internal_depth_m=internal,
            equalization_active=equalization_active) if current is not None else None
        state = evaluated.state if evaluated else 'WARNING'
        fresh = self.mode == 'quick' or sensor_status == 'OK'
        if fresh:
            self.last_valid_state = state
        else:
            # Losing telemetry must not downgrade an already observed DANGER.
            state = 'DANGER' if state == 'DANGER' or self.last_valid_state == 'DANGER' else 'WARNING'
        payload = {
            'mode': self.mode, 'prediction_source': 'mock', 'scenario': selected_scenario,
            'timestamp_utc': datetime.fromtimestamp(now, timezone.utc).isoformat(),
            'timestamp_source': 'host_receipt_time',
            'sensor_status': sensor_status, 'sensor_age_seconds': age,
            'received_sensor_depth_m': self.last_depth, 'current_depth_m': current,
            'received_internal_depth_m': self.last_internal_depth,
            'internal_depth_m': internal,
            'equalization_active': equalization_active,
            'forecast_status': evaluated.forecast_status if evaluated else forecast.status,
            'predicted_depths_m': forecast.predicted_depths_m,
            'forecast_origin_unix_seconds': forecast_origin,
            'history_status': 'BYPASSED' if self.mode == 'quick' else forecast.status,
            'history_bins': self.history_bins, 'observed_history_bins': self.observed_bins,
            'required_history_bins': 30, 'grid_origin_unix_seconds': self.grid_origin,
            'state': state,
            't_warning_min': evaluated.t_warning_min if evaluated and fresh else None,
            't_external_critical_min': evaluated.t_external_critical_min if evaluated and fresh else None,
            'pressure_status': evaluated.pressure_status if evaluated and fresh else sensor_status,
            'pressure_forecast_status': evaluated.pressure_forecast_status if evaluated and fresh else 'UNAVAILABLE',
            't_pressure_critical_if_internal_constant_min': (
                evaluated.t_pressure_critical_if_internal_constant_min if evaluated and fresh else None),
            'command': 'NONE',
        }
        for field in ('external_state', 'internal_state', 'pressure_state', 'depth_gap_m',
                      'equivalent_depth_m', 'bottom_pressure_difference_pa', 'pressure_direction'):
            payload[field] = getattr(evaluated, field) if evaluated and fresh else None
        payload['pressure_warning_equivalent_depth_m'] = self.risk.pressure_warning_equivalent_depth_m
        payload['pressure_critical_equivalent_depth_m'] = self.risk.pressure_critical_equivalent_depth_m
        return payload
