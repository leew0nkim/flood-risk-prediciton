"""Local review demonstrations; no device access, pretrained model, or source edits.

Run from the repository root:
    /opt/anaconda3/bin/python -m analysis.ab_review_20260923.reproduce
These describe current behavior, not the desired acceptance criteria.
"""
import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import warnings

import numpy as np
import torch

from meit_model.baselines import linear_forecast
from meit_model.electronics_io import ElectronicsSerial, SensorPacket
from meit_model.integration_test_modes import IntegrationSession
from meit_model.lstm import LSTMForecaster
from meit_model.predictor import ForecastPredictor, ForecastResult
from meit_model.risk import RiskEngine


def packet(depth):
    return SensorPacket(depth, None, False, False)


def main():
    config = json.loads(Path('configs/pipeline_config.json').read_text())
    results = {}

    session = IntegrationSession('accumulate', config, 0)
    first = session.update(packet(.5), 0)
    stale = session.update(None, 181)
    results['confirmed_danger_then_sensor_loss'] = {
        'initial_state': first['state'], 'after_181_seconds_state': stale['state'],
        'sensor_status': stale['sensor_status'], 'retained_depth_m': stale['current_depth_m'],
    }

    session = IntegrationSession('accumulate', config, 100)
    session.update(packet(.1), 101)
    try:
        session.update(None, 100.9)
    except ValueError as exc:
        results['wall_clock_rollback'] = {'exception': type(exc).__name__, 'message': str(exc)}

    old = datetime(2000, 1, 1, tzinfo=timezone.utc)
    forecast = ForecastResult('READY', [.1] * 15,
                              [old + timedelta(minutes=i) for i in range(1, 16)])
    risk = RiskEngine().evaluate(.1, forecast)
    results['old_ready_forecast'] = {
        'forecast_year': 2000, 'state': risk.state, 'forecast_status': risk.forecast_status,
    }

    class FakeSerial:
        def readline(self, size):
            return b'{"external_depth_m":1e40,"internal_depth_m":null,"evacuation_complete":false,"equalization_active":false}\n'

    transport = ElectronicsSerial('NOT_A_DEVICE', connection=FakeSerial())
    huge = transport.read_sensor_packet()
    session = IntegrationSession('accumulate', config, 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for second in range(0, 1801, 60):
            payload = session.update(huge, second)
    try:
        json.dumps(payload, allow_nan=False)
        serialization = 'accepted'
    except ValueError as exc:
        serialization = type(exc).__name__
    results['finite_input_nonfinite_ready_output'] = {
        'parser_accepted': huge is not None,
        'forecast_status': payload['forecast_status'],
        'all_predictions_finite': bool(np.isfinite(payload['predicted_depths_m']).all()),
        'serialization': serialization,
    }

    session = IntegrationSession('accumulate', config, 0)
    for second in range(0, 1801, 60):
        payload = session.update(packet(.29), second)
    results['float32_warning_boundary'] = {
        'input_m': .29, 'forecast_m': payload['predicted_depths_m'][0],
        'state': payload['state'], 't_warning_min': payload['t_warning_min'],
    }

    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix='meit-ab-review-') as temporary:
        model = LSTMForecaster(hidden_size=4)
        checkpoint = {
            'normalization_version': 1,
            'scaler': {'depth_unit': 'inches', 'mean_m': .1, 'std_m': .2},
            'config': {'input_size': 1, 'hidden_size': 4, 'num_layers': 1,
                       'horizon': 15, 'bin_seconds': 5},
            'model_state_dict': model.state_dict(),
        }
        path = Path(temporary) / 'synthetic_contract_mismatch.pt'
        torch.save(checkpoint, path)
        with contextlib.redirect_stdout(io.StringIO()):
            predictor = ForecastPredictor(path, device='cpu')
        result = predictor.predict([.1] * 30, origin_time=old)
        results['checkpoint_contract_mismatch'] = {
            'normalization_version': 1, 'depth_unit': 'inches', 'bin_seconds': 5,
            'loaded': True, 'prediction_status': result.status,
            'note': 'Synthetic untrained weights; this does not measure model accuracy.',
        }

    x = torch.full((1, 30, 1), .1)
    x[0, -5:, 0] = torch.tensor([.1, .1, .1, .3, .5])
    actual_times = np.array([-4., -1., 0.])
    actual_depths = np.array([.1, .3, .5])
    slope = np.polyfit(actual_times, actual_depths, 1)[0]
    results['baseline_specification_difference'] = {
        'last_five_bins_m': [.1, .1, .1, .3, .5],
        'observed_mask': [True, False, False, True, True],
        'actual_observation_minutes_relative_to_origin': actual_times.tolist(),
        'existing_plus_one_m': float(linear_forecast(x)[0, 0, 0]),
        'documented_observed_slope_last_observation_anchor_plus_one_m': float(.5 + slope),
        'note': 'The documented alternative is not asserted to be more accurate.',
    }

    directory = Path('data/processed')
    with np.load(directory / 'train.npz', allow_pickle=False) as archive:
        train_times = archive['origin_time_unix_seconds']
        train_events = archive['event_index']
    with np.load(directory / 'test_risk.npz', allow_pickle=False) as archive:
        risk_times = archive['origin_time_unix_seconds']
        risk_events = archive['event_index']
    manifest = json.loads((directory / 'split_manifest.json').read_text())
    sensors = {event['event_index']: event['sensor_id'] for event in manifest['events']}
    order = np.argsort(train_times)
    train_times, train_events = train_times[order], train_events[order]
    # Each information interval is (origin-30min, origin+15min].
    left = np.searchsorted(train_times, risk_times - 2700, side='right')
    right = np.searchsorted(train_times, risk_times + 2700, side='left')
    overlapping = right > left
    same_sensor_count = sum(any(
        sensors[int(event)] == sensors[int(risk_events[i])]
        for event in train_events[left[i]:right[i]]) for i in np.flatnonzero(overlapping))
    results['risk_holdout_time_overlap'] = {
        'risk_events_with_samples': len(np.unique(risk_events)),
        'risk_events_with_overlapping_training_sample_intervals': len(np.unique(risk_events[overlapping])),
        'event_indices': np.unique(risk_events[overlapping]).tolist(),
        'overlap_same_sensor_sample_count': same_sensor_count,
        'note': 'Cross-sensor temporal overlap; not proof of direct label leakage or one shared storm.',
    }
    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
