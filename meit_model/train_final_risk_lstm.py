"""Fit the fixed risk-aware LSTM on development data only; no test selection."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from meit_model.cv import load_development_data, load_scaler
from meit_model.fold_scaling import FoldScalerSource, rescale_data, indices_digest
from meit_model.lstm import LSTMForecaster
from meit_model.predictor import ForecastPredictor
from meit_model.train_lstm_cv import CVDataset, make_event_balanced_loader, set_seed
from meit_model.train_risk_lstm_cv import train_one_epoch_risk


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/final_model'))
    parser.add_argument('--epochs', type=int, default=19,
                        help='Final deployment recipe requires 19 epochs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.epochs != 19 or args.seed != 42:
        parser.error('Final deployment recipe requires --epochs 19 --seed 42. '
                     'Use a separate script for smoke experiments.')
    # Refuse reuse, including partially completed runs, to preserve provenance.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory is not empty; choose a new --output-dir')

    config = dict(input_size=1, input_length=30, hidden_size=64, num_layers=1,
                  horizon=15, batch_size=128, learning_rate=.001, gradient_clip=1.,
                  critical_depth_m=.300, underprediction_weight=2.,
                  max_epochs=args.epochs, seed=args.seed, normalization_version=2)
    policy = dict(forecast_warning_depth_m=.295,
                  pressure_critical_equivalent_depth_m=.300,
                  required_lead_time_min=5, danger_horizon_min=5)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_development_data(args.data_dir)
    indices = np.arange(len(data.X), dtype=np.int64)
    source = FoldScalerSource(args.data_dir)
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
        epoch_selection_rule='CV v2 seeds 42/43/44 best-epoch median 18.5 rounded to 19',
        cv_best_epochs=[39, 15, 17, 21, 28, 15, 14, 27, 33, 11, 17, 20],
        fixed_epoch_rule_applied=args.epochs == 19,
        deployment_recipe_applied=args.epochs == 19 and args.seed == 42,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'training_config.json',
               dict(config=config, policy=policy, scaler=scaler, **provenance))
    print('FINAL RISK-AWARE LSTM TRAINING', flush=True)
    print(f'Device: {device}; epochs: {args.epochs}; seed: {args.seed}', flush=True)
    print(f'Development windows: {len(indices)}; scaler: {scaler}', flush=True)
    print('No validation selection or sealed test evaluation.', flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch_risk(
            model, loader, optimizer, scaler, device,
            config['gradient_clip'], config['critical_depth_m'], config['underprediction_weight'])
        if not math.isfinite(loss):
            raise RuntimeError('Nonfinite training loss; no final checkpoint saved')
        history.append(dict(epoch=epoch, risk_loss=loss))
        write_json(args.output_dir / 'history.json', history)
        print(f'Epoch {epoch:02d}/{args.epochs} | risk loss={loss:.8f}', flush=True)

    checkpoint_path = args.output_dir / 'final_model.pt'
    checkpoint = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                      scaler=scaler, config=config, policy=policy, normalization_version=2,
                      epoch=args.epochs, training_epochs=args.epochs, seed=args.seed,
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
    write_json(args.output_dir / 'training_summary.json',
               dict(**provenance, training_epochs=args.epochs, seed=args.seed,
                    checkpoint=str(checkpoint_path), predictor_smoke_status=result.status))
    print(f'FINAL TRAINING COMPLETE: {checkpoint_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
