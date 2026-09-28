"""Descriptive validation-only comparison of DANGER cutoffs; no LSTM claim.

Run from the repository root:
    python -m analysis.danger_horizon_20260923.compare
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from meit_data.torch_data import ForecastDataset, depth_to_m
from meit_model.baselines import linear_forecast, persistence_forecast
from meit_model.evaluate_event_level import evaluate_events
from meit_model.predictor import ForecastResult
from meit_model.risk import RiskEngine


def main():
    root = Path(__file__).resolve().parents[2]
    dataset = ForecastDataset(root / 'data/processed', 'validation')
    arrays, scaler = dataset.arrays, dataset.scaler
    current = depth_to_m(arrays['X'][:, -1, 0], scaler).numpy().astype(np.float64)
    # Stored float32 zero depths inverse-transform to -7.45e-9 m here.
    assert current.min() >= -1e-7
    roundoff_count = int((current < 0).sum())
    current = np.maximum(current, 0)
    actual = depth_to_m(arrays['Y'][:, :, 0], scaler).numpy().astype(np.float64)
    mask = arrays['Y_mask'][:, :, 0].numpy()
    ids = arrays['event_index'].numpy()
    times = arrays['origin_time_unix_seconds'].numpy()
    critical, warning = .30, .29
    horizons = [1, 3, 5]
    eligible = (current < critical) & mask.all(axis=1)
    positive = (actual >= critical).any(axis=1)
    payload = {
        'scope': 'Validation-only descriptive baselines; not LSTM or final test performance',
        'current_critical_m': critical,
        'forecast_warning_m': warning,
        'sample_count': len(ids),
        'roundoff_only_current_depths_clamped_to_zero': roundoff_count,
        'event_count': len(np.unique(ids)),
        'fixed_future_observation_horizon_min': 15,
        'complete_precritical_windows': int(eligible.sum()),
        'positive_windows': int((eligible & positive).sum()),
        'negative_windows': int((eligible & ~positive).sum()),
        'source_hashes': {
            name: hashlib.sha256((root / 'data/processed' / name).read_bytes()).hexdigest()
            for name in ['validation.npz', 'scaler.json', 'split_manifest.json']
        },
        'limitations': [
            'Only represented validation origins with READY forecasts are replayed.',
            'First observable future crossing is not necessarily the first raw-event crossing.',
            'Event warnings searched only in the preceding 15 minutes, as in the existing evaluator.',
            'No dry-period stream or warmup replay; false alarms are overlapping window counts.',
            'Five-minute advance red-warning recall is diagnostic, not the existing yellow-or-red requirement.',
            'No final holdout is loaded and no model or operating threshold is changed.',
        ],
        'models': {},
    }
    for name, model in [('linear_5point', linear_forecast), ('persistence', persistence_forecast)]:
        with torch.no_grad():
            prediction = depth_to_m(model(arrays['X']), scaler, clip_negative=True)[:, :, 0].numpy().astype(np.float64)
        base_events = evaluate_events(ids, times, current, actual, mask, prediction, warning)
        states, results = {}, {}
        for horizon in horizons:
            engine = RiskEngine(danger_horizon_min=horizon)
            states[horizon] = np.array([
                # Historical study has external data only; do not invent indoor observations.
                engine.evaluate(float(depth), ForecastResult('READY', row.tolist(), None)).external_state
                for depth, row in zip(current, prediction)
            ])
            danger = states[horizon] == 'DANGER'
            expected = (current >= critical) | (prediction[:, :horizon] >= warning).any(axis=1)
            assert np.array_equal(danger, expected)
            event_rows = []
            for event in base_events:
                crossing, event_id = event['actual_crossing_time'], event['event_index']
                candidates = ((ids == event_id) & (current < critical)
                              & (times >= crossing - 15 * 60) & (times <= crossing))
                alarms = times[candidates & danger]
                lead = float((crossing - alarms.min()) / 60) if len(alarms) else None
                metadata = dataset.events[event_id]
                event_rows.append({
                    'event_index': event_id, 'sensor_id': metadata['sensor_id'],
                    'event_start_utc': metadata['start_utc'],
                    'actual_crossing_time': crossing, 'lead_time_min': lead,
                })
            leads = [r['lead_time_min'] for r in event_rows if r['lead_time_min'] is not None]
            tp = int((eligible & positive & danger).sum())
            fp = int((eligible & ~positive & danger).sum())
            fn = int((eligible & positive & ~danger).sum())
            tn = int((eligible & ~positive & ~danger).sum())
            results[str(horizon)] = {
                'crossing_event_count': len(event_rows),
                'detected_event_count': len(leads),
                'median_lead_min': float(np.median(leads)) if leads else None,
                'min_lead_min': min(leads) if leads else None,
                'timely_event_counts': {str(k): sum(t >= k for t in leads) for k in horizons},
                'window_tp': tp, 'window_fp': fp, 'window_fn': fn, 'window_tn': tn,
                'window_precision': tp / (tp + fp) if tp + fp else None,
                'window_recall': tp / (tp + fn) if tp + fn else None,
                'window_f2': 5 * tp / (5 * tp + 4 * fn + fp)
                    if 5 * tp + 4 * fn + fp else None,
                'window_fpr': fp / (fp + tn) if fp + tn else None,
                'window_false_alarm_ratio': fp / (fp + tp) if fp + tp else None,
                'all_danger_origins': int(danger.sum()),
                'event_rows': event_rows,
            }
        assert np.array_equal(states[1] != 'SAFE', states[3] != 'SAFE')
        assert np.array_equal(states[3] != 'SAFE', states[5] != 'SAFE')
        added = (states[5] == 'DANGER') & (states[3] != 'DANGER')
        results['three_to_five'] = {
            'additional_danger_origins_all': int(added.sum()),
            'additional_danger_origins_complete_precritical': int((eligible & added).sum()),
            'additional_true_alarm_origins': int((eligible & added & positive).sum()),
            'additional_false_alarm_origins': int((eligible & added & ~positive).sum()),
            'yellow_or_red_states_unchanged': True,
        }
        payload['models'][name] = results
    output = Path(__file__).with_name('results.json')
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({**{k: v for k, v in payload.items() if k != 'models'},
                      'models': {m: {h: {k: v for k, v in row.items() if k != 'event_rows'}
                                     for h, row in rows.items()}
                                 for m, rows in payload['models'].items()}}, indent=2))


if __name__ == '__main__':
    main()
