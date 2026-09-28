"""Architecture/data diagnostics only; no training or held-out test scoring.

Run from the repository root with:
  /opt/anaconda3/bin/python -m analysis.lstm_architecture_review_20260925.diagnostics
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from meit_model.lstm import LSTMForecaster
from meit_model.risk_loss import risk_aware_masked_mae


def main():
    torch.manual_seed(42)
    directory = Path('data/processed')
    scaler = json.loads((directory / 'scaler.json').read_text())
    result = {'purpose': 'structural_diagnostics_not_trained_model_comparison',
              'architecture': {}, 'development_splits': {}}
    for features, hidden, layers in ((1, 64, 1), (3, 64, 1), (3, 32, 1), (3, 64, 2)):
        model = LSTMForecaster(input_size=features, hidden_size=hidden, num_layers=layers)
        output = model(torch.zeros(2, 30, features))
        output.abs().mean().backward()
        result['architecture'][f'features_{features}_hidden_{hidden}_layers_{layers}'] = {
            'parameters': sum(p.numel() for p in model.parameters()),
            'output_shape': list(output.shape),
            'finite_outputs': bool(torch.isfinite(output).all()),
            'finite_gradients': all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                    for p in model.parameters()),
        }

    quantile_core = LSTMForecaster(input_size=3, hidden_size=32, horizon=45)
    result['candidate_A_quantile_parameter_count'] = sum(p.numel() for p in quantile_core.parameters())
    # Candidate B's skip connection adds no learned parameters.
    residual_core = LSTMForecaster(input_size=3, hidden_size=64)
    torch.nn.init.zeros_(residual_core.output_layer.weight)
    torch.nn.init.zeros_(residual_core.output_layer.bias)
    features = torch.rand(4, 30, 3)
    features[:, :, 1] = (features[:, :, 1] > .5).float()
    residual_output = residual_core(features) + features[:, -1:, 0:1]
    (residual_output - (residual_output.detach() + .1)).abs().mean().backward()
    result['candidate_B_untrained_structural_check'] = {
        'parameters': sum(p.numel() for p in residual_core.parameters()),
        'output_shape': list(residual_output.shape),
        'zero_head_initial_prediction_exactly_persistence': bool(torch.equal(
            residual_output, features[:, -1:, 0:1].expand(-1, 15, -1))),
        'first_backward_encoder_gradients_are_zero': all(
            bool((p.grad == 0).all()) for p in residual_core.lstm.parameters()),
        'first_backward_head_has_nonzero_gradient': bool(
            (residual_core.output_layer.weight.grad != 0).any()),
        'meaning': 'Shape and initialization invariants only; no accuracy or safety claim.',
    }

    for split in ('train', 'validation'):
        with np.load(directory / f'{split}.npz', allow_pickle=False) as archive:
            x = archive['X']
            observed = archive['X_observed'].reshape(-1, 30)
            age = archive['X_age_seconds'].reshape(-1, 30)
            targets_m = archive['Y'][:, :, 0].astype(float) * scaler['std_m'] + scaler['mean_m']
            target_mask = archive['Y_mask'][:, :, 0]
            events = archive['event_index']
        current_m = x[:, -1, 0].astype(float) * scaler['std_m'] + scaler['mean_m']
        high = target_mask & (targets_m >= .3)
        onset = ((current_m < .3) & observed[:, -1] & target_mask.all(axis=1)
                 & (targets_m >= .3).any(axis=1))
        first_index, different_mask_groups = {}, set()
        example = None
        for index, row in enumerate(x):
            key = row.tobytes()
            previous = first_index.setdefault(key, index)
            if not np.array_equal(observed[previous], observed[index]):
                different_mask_groups.add(key)
                if example is None:
                    example = {
                        'sample_indices': [previous, index],
                        'observed_masks': [observed[previous].tolist(), observed[index].tolist()],
                        'identical_depth_input': True,
                    }
        result['development_splits'][split] = {
            'windows': len(x), 'events': len(np.unique(events)),
            'windows_with_imputed_bins': int((~observed).any(axis=1).sum()),
            'fraction_windows_with_imputed_bins': float((~observed).any(axis=1).mean()),
            'fraction_last_bin_imputed': float((~observed[:, -1]).mean()),
            'age_seconds_percentiles_0_50_90_99_100': np.percentile(age, [0, 50, 90, 99, 100]).tolist(),
            'observed_target_positions': int(target_mask.sum()),
            'target_positions_at_least_030_m': int(high.sum()),
            'fraction_high_target_positions': float(high.sum() / target_mask.sum()),
            'events_with_fully_observed_onset_windows': len(np.unique(events[onset])),
            'identical_depth_input_groups_with_different_observed_masks': len(different_mask_groups),
            'example_identical_input_different_mask': example,
        }

    # A counterexample for the objectives, not outputs of fitted models.
    target = torch.tensor([.1] * 99 + [.35]).view(100, 1, 1).repeat(1, 15, 1)
    observed = torch.ones_like(target, dtype=torch.bool)
    p = target.clone()
    p[-1] = .15
    q = target.clone()
    q[:-1] = .1016
    q[-1] = .30
    result['synthetic_checkpoint_ranking_counterexample'] = {}
    for name, prediction in (('P_lower_overall_MAE', p), ('Q_lower_risk_loss', q)):
        result['synthetic_checkpoint_ranking_counterexample'][name] = {
            'overall_MAE_m': float((prediction - target).abs().mean()),
            'risk_weighted_loss_identity_scaler': float(risk_aware_masked_mae(
                prediction, target, observed, {'mean_m': 0., 'std_m': 1.},
                underprediction_weight=2.)),
            'high_water_underprediction_m': float((target[-1] - prediction[-1]).mean()),
        }

    sources = [Path('meit_model/lstm.py'), Path('meit_model/train_lstm_cv.py'),
               Path('meit_model/train_risk_lstm_cv.py'), Path('meit_model/risk_loss.py'),
               directory / 'train.npz', directory / 'validation.npz', directory / 'scaler.json']
    result['sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    result['torch_version'] = str(torch.__version__)
    result['trained_checkpoints_found'] = [str(path) for path in Path('.').rglob('*.pt')]
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
