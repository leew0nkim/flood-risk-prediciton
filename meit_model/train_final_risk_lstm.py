"""Fit the fixed risk-aware LSTM on development data only; no test selection."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
import torch

from meit_data.pipeline import sha256
from meit_model.cv import load_development_data, load_scaler
from meit_model.fold_scaling import FoldScalerSource, rescale_data, indices_digest
from meit_model.lstm import LSTMForecaster
from meit_model.predictor import ForecastPredictor
from meit_model.risk_policy import load_policy_decision, read_json
from meit_model.train_lstm_cv import CVDataset, make_event_balanced_loader, set_seed
from meit_model.train_risk_lstm_cv import train_one_epoch_risk


REQUIRED_CV_SEEDS = (42, 43, 44)


def json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Nonfinite float in JSON')
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                               default=json_default), encoding='utf-8')


def derive_training_recipe(cv_summary_paths, policy_decision_path,
                           window_calibration_path, event_evaluation_path, fingerprint):
    """Build the final recipe only from artifacts for the current dataset."""
    summaries = [read_json(path) for path in cv_summary_paths]
    seeds = []
    best_epochs = []
    cv_split_hashes = []
    for path, summary in zip(cv_summary_paths, summaries):
        if summary.get('schema_version') != 2:
            raise ValueError(f'Legacy CV summary is not accepted: {path}')
        if summary.get('data_fingerprint') != fingerprint:
            raise ValueError(f'CV summary was produced from different data: {path}')
        seed = summary.get('seed')
        if seed in seeds:
            raise ValueError(f'Duplicate CV seed {seed}: {path}')
        seeds.append(seed)
        split_hash = summary.get('cv_split_summary_sha256')
        if not isinstance(split_hash, str) or len(split_hash) != 64:
            raise ValueError(f'CV summary has no split-manifest SHA-256: {path}')
        cv_split_hashes.append(split_hash)
        folds = summary.get('folds')
        if not isinstance(folds, list) or len(folds) != 4:
            raise ValueError(f'CV summary must contain four folds: {path}')
        for fold in folds:
            epoch = fold.get('best_epoch') if isinstance(fold, dict) else None
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
                raise ValueError(f'Invalid best epoch in CV summary: {path}')
            best_epochs.append(epoch)
    if tuple(sorted(seeds)) != REQUIRED_CV_SEEDS:
        raise ValueError('Final recipe requires CV summaries for seeds 42, 43 and 44')
    if len(set(cv_split_hashes)) != 1:
        raise ValueError('CV seed runs used different fold definitions')

    decision = load_policy_decision(
        policy_decision_path, window_calibration_path, event_evaluation_path, fingerprint)
    threshold = decision['forecast_warning_depth_m']

    # Round .5 upward. Python round() uses bankers' rounding and would turn
    # an 18.5-epoch median into 18 instead of the intended 19.
    epochs = int(math.floor(float(median(best_epochs)) + .5))
    return {
        'epochs': epochs,
        'forecast_warning_depth_m': float(threshold),
        'cv_seeds': sorted(seeds),
        'cv_best_epochs': best_epochs,
        'cv_split_summary_sha256': cv_split_hashes[0],
        'cv_summary_paths': [str(path) for path in cv_summary_paths],
        'cv_summary_sha256': [sha256(path) for path in cv_summary_paths],
        'policy_decision_path': str(policy_decision_path),
        'policy_decision_sha256': sha256(policy_decision_path),
        'window_calibration_path': str(window_calibration_path),
        'window_calibration_sha256': sha256(window_calibration_path),
        'event_evaluation_path': str(event_evaluation_path),
        'event_evaluation_sha256': sha256(event_evaluation_path),
        'epoch_selection_rule': 'median of 12 validation best epochs, rounded .5 upward',
        'warning_threshold_selection_rule': decision['selection_basis'],
        'policy_decision': decision,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/final_model'))
    parser.add_argument('--cv-summary', type=Path, action='append', required=True,
                        help='Repeat for the seed 42, 43 and 44 CV summaries')
    parser.add_argument('--policy-decision', type=Path, required=True,
                        help='Locked, human-reviewed warning-threshold policy JSON')
    parser.add_argument('--window-calibration', type=Path, required=True)
    parser.add_argument('--event-evaluation', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.seed != 42:
        parser.error('Final deployment seed must be 42. Use a separate script for smoke experiments.')
    # Refuse reuse, including partially completed runs, to preserve provenance.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory is not empty; choose a new --output-dir')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_development_data(args.data_dir)
    indices = np.arange(len(data.X), dtype=np.int64)
    source = FoldScalerSource(args.data_dir)
    try:
        recipe = derive_training_recipe(
            args.cv_summary, args.policy_decision, args.window_calibration,
            args.event_evaluation, source.fingerprint)
    except ValueError as exc:
        parser.error(str(exc))
    epochs = recipe['epochs']
    config = dict(input_size=1, input_length=30, hidden_size=64, num_layers=1,
                  horizon=15, batch_size=128, learning_rate=.001, gradient_clip=1.,
                  critical_depth_m=.300, underprediction_weight=2.,
                  max_epochs=epochs, seed=args.seed, normalization_version=2)
    policy = dict(recipe['policy_decision'])
    policy['danger_horizon_min'] = 5
    scaler = source.fit(indices)
    scaler['scope'] = 'full_development_training_inputs_only'
    scaled = rescale_data(data, load_scaler(args.data_dir), scaler)
    dataset = CVDataset(scaled, indices)
    loader = make_event_balanced_loader(dataset, batch_size=config['batch_size'], seed=args.seed)
    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=1, horizon=15).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    provenance = dict(
        training_scope='full_development_train_plus_validation',
        data_fingerprint=source.fingerprint, train_indices_sha256=indices_digest(indices),
        development_window_count=len(indices), sealed_test_used=False,
        epoch_selection_rule=recipe['epoch_selection_rule'],
        warning_threshold_selection_rule=recipe['warning_threshold_selection_rule'],
        cv_seeds=recipe['cv_seeds'], cv_best_epochs=recipe['cv_best_epochs'],
        cv_split_summary_sha256=recipe['cv_split_summary_sha256'],
        cv_summary_paths=recipe['cv_summary_paths'], cv_summary_sha256=recipe['cv_summary_sha256'],
        policy_decision_path=recipe['policy_decision_path'],
        policy_decision_sha256=recipe['policy_decision_sha256'],
        window_calibration_path=recipe['window_calibration_path'],
        window_calibration_sha256=recipe['window_calibration_sha256'],
        event_evaluation_path=recipe['event_evaluation_path'],
        event_evaluation_sha256=recipe['event_evaluation_sha256'],
        deployment_recipe_applied=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'training_config.json',
               dict(config=config, policy=policy, scaler=scaler, **provenance))
    print('FINAL RISK-AWARE LSTM TRAINING', flush=True)
    print(f'Device: {device}; epochs: {epochs}; seed: {args.seed}', flush=True)
    print(f'Development windows: {len(indices)}; scaler: {scaler}', flush=True)
    print('No validation selection or sealed test evaluation.', flush=True)
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch_risk(
            model, loader, optimizer, scaler, device,
            config['gradient_clip'], config['critical_depth_m'], config['underprediction_weight'])
        if not math.isfinite(loss):
            raise RuntimeError('Nonfinite training loss; no final checkpoint saved')
        history.append(dict(epoch=epoch, risk_loss=loss))
        write_json(args.output_dir / 'history.json', history)
        print(f'Epoch {epoch:02d}/{epochs} | risk loss={loss:.8f}', flush=True)

    checkpoint_path = args.output_dir / 'final_model.pt'
    checkpoint = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                      scaler=scaler, config=config, policy=policy, normalization_version=2,
                      epoch=epochs, training_epochs=epochs, seed=args.seed,
                      loss='risk_aware_asymmetric_mae_valid_mean_v2', **provenance)
    # Check predictor compatibility before publishing the final filename.
    pending = args.output_dir / 'pending_model.pt'
    torch.save(checkpoint, pending)
    predictor = ForecastPredictor(pending)
    example_m = (scaled.X[0, :, 0].astype(np.float64) * scaler['std_m'] + scaler['mean_m']).tolist()
    result = predictor.predict(example_m)
    if result.status != 'READY':
        raise RuntimeError(f'Saved predictor smoke failed: {result.status}; {result.message}')
    pending.replace(checkpoint_path)
    checkpoint_sha256 = sha256(checkpoint_path)
    write_json(args.output_dir / 'training_summary.json',
               dict(**provenance, training_epochs=epochs, seed=args.seed,
                    checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha256,
                    predictor_smoke_status=result.status))
    print(f'FINAL TRAINING COMPLETE: {checkpoint_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
