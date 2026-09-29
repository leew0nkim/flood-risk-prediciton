"""Event recall within 15 minutes before the first observable future crossing.

This evaluates positive events represented by validation windows, not full raw
flood histories. Missing observations can hide an earlier crossing. Negative
and unobservable events are excluded, so this is not an event-level FPR.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from meit_data.pipeline import sha256
from meit_model.calibrate_risk_warning import collect_predictions
from meit_model.cv import load_development_data, load_scaler
from meit_model.evaluate_lstm_cv import load_checkpoint_model
from meit_model.fold_scaling import checkpoint_data, data_fingerprint

CRITICAL_DEPTH_M = 0.300
HORIZON_MINUTES = 15


def evaluate_events(event_ids, origin_times, current_depth_m, target_m,
                    target_observed, prediction_m, warning_threshold_m):
    """Inputs are metre arrays; targets/predictions/mask have shape (N, 15).

    Matches the supplied protocol: use observed future labels from origins
    below critical; look for alarms at origins in [crossing-15min, crossing].
    A zero-minute warning counts as detected, and is reported separately.
    """
    event_ids, origin_times, current_depth_m = map(
        np.asarray, (event_ids, origin_times, current_depth_m))
    target_m, target_observed, prediction_m = map(
        np.asarray, (target_m, target_observed, prediction_m))
    n = len(event_ids)
    if any(a.shape != (n,) for a in (event_ids, origin_times, current_depth_m)):
        raise ValueError('Event IDs, origins and current depths must have shape (N,)')
    if any(a.shape != (n, 15) for a in (target_m, target_observed, prediction_m)):
        raise ValueError('Targets, masks and predictions must have shape (N,15)')
    if target_observed.dtype != np.bool_:
        raise TypeError('Target mask must be boolean')
    if not (np.issubdtype(event_ids.dtype, np.integer)
            and np.issubdtype(origin_times.dtype, np.integer)):
        raise TypeError('Event IDs and Unix seconds must be integers')
    if not (np.isfinite(current_depth_m).all() and np.isfinite(prediction_m).all()
            and np.isfinite(target_m[target_observed]).all()):
        raise ValueError('Nonfinite observed depths or predictions')
    if not np.isfinite(warning_threshold_m) or warning_threshold_m <= 0:
        raise ValueError('Warning threshold must be positive and finite')
    if n and len(np.unique(np.column_stack((event_ids, origin_times)), axis=0)) != n:
        raise ValueError('Duplicate event/origin samples')

    results = []
    offsets = np.arange(1, 16, dtype=np.int64) * 60
    for event_id in np.unique(event_ids):
        idx = np.flatnonzero(event_ids == event_id)
        idx = idx[np.argsort(origin_times[idx], kind='stable')]
        eligible = idx[current_depth_m[idx] < CRITICAL_DEPTH_M]
        crossings = target_observed[eligible] & (target_m[eligible] >= CRITICAL_DEPTH_M)
        if not crossings.any():
            continue
        times = origin_times[eligible, None] + offsets
        crossing = int(times[crossings].min())
        candidates = eligible[(origin_times[eligible] >= crossing - 15 * 60)
                              & (origin_times[eligible] <= crossing)]
        alarms = candidates[(prediction_m[candidates] >= warning_threshold_m).any(axis=1)]
        first = int(origin_times[alarms].min()) if len(alarms) else None
        results.append({
            'event_index': int(event_id), 'actual_crossing_time': crossing,
            'first_alarm_time': first,
            'lead_time_min': (crossing - first) / 60.0 if first is not None else None,
            'detected': first is not None,
            'warning_threshold_m': float(warning_threshold_m),
            'eligible_pre_crossing_window_count': int(len(candidates)),
        })
    return results


def summarize_events(results, required_lead_time_min=None):
    if required_lead_time_min is not None and (
            not np.isfinite(required_lead_time_min) or not 0 <= required_lead_time_min <= 15):
        raise ValueError('Required lead time must be finite and within 0..15 minutes')
    operational = None if required_lead_time_min is None else sum(
        r['detected'] and r['lead_time_min'] >= required_lead_time_min for r in results)

    detected = [r for r in results if r['detected']]
    lead = np.array([r['lead_time_min'] for r in detected], dtype=float)
    return {
        'required_lead_time_min': required_lead_time_min,
        'operational_timely_event_count': operational,
        'operational_event_recall': operational / len(results)
            if operational is not None and results else None,
        'event_recall': len(detected) / len(results) if results else None,
        'crossing_event_count': len(results),
        'detected_event_count': len(detected),
        'missed_event_count': len(results) - len(detected),
        'event_timely_recall': len(detected) / len(results) if results else None,
        'positive_lead_event_count': int((lead > 0).sum()),
        'zero_lead_event_count': int((lead == 0).sum()),
        **{name + '_lead_time_min': float(function(lead)) if lead.size else None
           for name, function in [('mean', np.mean), ('median', np.median),
                                  ('min', np.min), ('max', np.max)]},
    }


def reaggregate_report(payload, required_lead_time_min):
    """Recompute summaries from saved event records without model inference."""
    grouped = {}
    for fold in payload['folds']:
        for result in fold['results']:
            rows = result['events']
            for row in rows:
                row['required_lead_time_min'] = required_lead_time_min
                row['operationally_timely'] = (
                    bool(row['detected'] and row['lead_time_min'] >= required_lead_time_min)
                    if required_lead_time_min is not None else None)
            result['summary'] = summarize_events(rows, required_lead_time_min)
            grouped.setdefault(result['warning_threshold_m'], []).extend(rows)
    payload['required_lead_time_min'] = required_lead_time_min
    for rows in grouped.values():
        if len({r['event_index'] for r in rows}) != len(rows):
            raise ValueError('Duplicate events across folds')
    payload['pooled'] = [
        {'warning_threshold_m': threshold,
         'summary': summarize_events(rows, required_lead_time_min)}
        for threshold, rows in grouped.items()]
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--cv-dir', type=Path, default=Path('artifacts/cv'))
    parser.add_argument('--model-dir', type=Path,
                        help='Checkpoint root (default: artifacts/risk_lstm_cv_v2)')
    parser.add_argument('--input-results', type=Path, help='Reaggregate a saved event JSON without inference')
    parser.add_argument('--required-lead-time-min', type=float, default=5,
                        help='Minimum operational warning lead in minutes (default: 5)')
    parser.add_argument('--output', type=Path,
                        help='Result JSON (default: <model-dir>/event_evaluation.json)')
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--warning-thresholds', type=float, nargs='+',
                        default=[.300, .295, .290, .285, .280])
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.required_lead_time_min is not None and (
            not np.isfinite(args.required_lead_time_min) or not 0 <= args.required_lead_time_min <= 15):
        parser.error('required-lead-time-min must be within 0..15')
    if args.input_results:
        if args.model_dir or args.output is None:
            parser.error('Reaggregation requires --output and no --model-dir')
        payload = json.loads(args.input_results.read_text(encoding='utf-8'))
        payload = reaggregate_report(payload, args.required_lead_time_min)
        payload['reaggregated_from'] = str(args.input_results)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(payload['pooled'], indent=2))
        print('Saved:', args.output)
        return
    if args.model_dir is None:
        args.model_dir = Path('artifacts/risk_lstm_cv_v2')
    if args.batch_size < 1 or len(set(args.folds)) != len(args.folds):
        parser.error('Use a positive batch size and distinct folds')
    if any(not np.isfinite(t) or t <= 0 for t in args.warning_thresholds):
        parser.error('Thresholds must be positive and finite')
    data = load_development_data(args.data_dir)
    # Only undo the on-disk transform with the source scaler. All model input
    # normalization and prediction/target inverse transforms use the checkpoint.
    source_scaler = load_scaler(args.data_dir)
    fingerprint = data_fingerprint(args.data_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pooled = {t: [] for t in args.warning_thresholds}
    folds, seen_events, model_evidence = [], set(), []
    for number in args.folds:
        with np.load(args.cv_dir / f'fold_{number}.npz', allow_pickle=False) as fold:
            indices = fold['val_indices']
        if indices.ndim != 1 or not len(indices) or indices.min() < 0 or indices.max() >= len(data.X):
            raise ValueError('Invalid validation indices')
        event_set = set(data.event_index[indices].tolist())
        if event_set & seen_events:
            raise ValueError('Validation events occur in multiple folds; cannot pool event recall')
        seen_events.update(event_set)
        checkpoint_path = args.model_dir / f'fold_{number}/best_model.pt'
        model, checkpoint = load_checkpoint_model(checkpoint_path, device)
        model_evidence.append({'fold': number, 'checkpoint': str(checkpoint_path),
                               'sha256': sha256(checkpoint_path)})
        if checkpoint.get('fold') != number:
            raise ValueError('Checkpoint fold number mismatch')
        fold_data, scaler = checkpoint_data(data, source_scaler, checkpoint, fingerprint, indices)
        arrays = collect_predictions(model, fold_data, indices, scaler, device, args.batch_size)
        arrays = {k: v.numpy() for k, v in arrays.items()}
        comparisons = []
        for threshold in args.warning_thresholds:
            events = evaluate_events(data.event_index[indices], data.origin_time[indices],
                                     arrays['current_m'], arrays['actual_m'], arrays['mask'],
                                     arrays['lstm_m'], threshold)
            for event in events:
                event['fold'] = number
            pooled[threshold].extend(events)
            summary = summarize_events(events, args.required_lead_time_min)
            comparisons.append({'warning_threshold_m': threshold, 'summary': summary, 'events': events})
            print(f'Fold {number}, threshold {threshold:.3f}: {summary}')
        folds.append({'fold': number, 'validation_event_count': len(event_set),
                      'scaler': scaler, 'checkpoint_epoch': checkpoint['epoch'], 'results': comparisons})
    output = args.output or args.model_dir / 'event_evaluation.json'
    payload = {
        'schema_version': 2,
        'report_type': 'event_evaluation',
        'actual_critical_depth_m': CRITICAL_DEPTH_M,
        'cv_dir': str(args.cv_dir), 'model_dir': str(args.model_dir),
        'model_evidence': model_evidence, 'data_fingerprint': fingerprint,
        'protocol': 'first observed future crossing in validation windows; alarm origin in preceding 15 minutes including crossing; current depth below critical',
        'limitations': ['Not full raw-event first crossings; missing/unrepresented labels may hide earlier crossings.',
                        'Non-crossing and unobservable events excluded; no event-level false alarm rate.',
                        'Zero-minute lead counts as detected, not positive advance warning.'],
        'folds': folds,
        'pooled': [{'warning_threshold_m': t, 'summary': summarize_events(rows, args.required_lead_time_min)}
                   for t, rows in pooled.items()],
    }
    payload = reaggregate_report(payload, args.required_lead_time_min)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(payload['pooled'], indent=2))
    print('Saved:', output)


if __name__ == '__main__':
    main()
