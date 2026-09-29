# Risk-aware LSTM selection workflow

All selection uses development data only. Do not run or inspect either sealed
test split until the CV epoch recipe and warning policy are locked.

1. Generate CV folds in a new empty directory. `cv_summary.json` records the
   preprocessing fingerprint and each fold file's SHA-256.
2. Train all four folds for seeds 42, 43 and 44. The trainer rejects fold files
   whose fingerprint or SHA-256 does not match `cv_summary.json`.
3. Run window calibration for the candidate thresholds. This report contains
   window recall, timely recall, FPR, precision and alarm rate, but does not
   select a threshold.
4. Run event evaluation with `--required-lead-time-min 5` for the same
   candidates. The primary metric is `operational_event_recall`: the fraction
   of represented crossing events warned at least five minutes in advance.
5. Review operational event recall together with window FPR, precision and
   alarm rate. Lock an explicitly chosen threshold with
   `finalize_risk_policy`; there is no automatic ranking rule.
6. Final training requires the three seed summaries, locked decision, window
   report and event report. It verifies their dataset fingerprints and hashes,
   derives epochs from the 12 validation best epochs and records the final
   checkpoint SHA-256.

Example commands, with a new run name for every attempt:

```bash
python -m meit_model.cv \
  --data-dir data/processed \
  --output-dir artifacts/cv_canonical_v2

python -m meit_model.train_risk_lstm_cv --data-dir data/processed --cv-dir artifacts/cv_canonical_v2 --output-dir artifacts/risk_cv_seed42 --seed 42
python -m meit_model.train_risk_lstm_cv --data-dir data/processed --cv-dir artifacts/cv_canonical_v2 --output-dir artifacts/risk_cv_seed43 --seed 43
python -m meit_model.train_risk_lstm_cv --data-dir data/processed --cv-dir artifacts/cv_canonical_v2 --output-dir artifacts/risk_cv_seed44 --seed 44

python -m meit_model.calibrate_risk_warning \
  --data-dir data/processed --cv-dir artifacts/cv_canonical_v2 \
  --model-dir artifacts/risk_cv_seed42 \
  --output artifacts/review/window_calibration.json

python -m meit_model.evaluate_event_level \
  --data-dir data/processed --cv-dir artifacts/cv_canonical_v2 \
  --model-dir artifacts/risk_cv_seed42 --required-lead-time-min 5 \
  --warning-thresholds .300 .295 .290 .285 .280 \
  --output artifacts/review/event_evaluation_5min.json

python -m meit_model.finalize_risk_policy \
  --data-dir data/processed \
  --window-calibration artifacts/review/window_calibration.json \
  --event-evaluation artifacts/review/event_evaluation_5min.json \
  --selected-threshold SELECTED_VALUE \
  --selection-note "Reviewed 5-minute event recall, FPR, precision and alarm rate." \
  --output artifacts/review/locked_policy.json

python -m meit_model.train_final_risk_lstm \
  --data-dir data/processed \
  --cv-summary artifacts/risk_cv_seed42/training_summary.json \
  --cv-summary artifacts/risk_cv_seed43/training_summary.json \
  --cv-summary artifacts/risk_cv_seed44/training_summary.json \
  --window-calibration artifacts/review/window_calibration.json \
  --event-evaluation artifacts/review/event_evaluation_5min.json \
  --policy-decision artifacts/review/locked_policy.json \
  --output-dir artifacts/final_model_canonical_v2
```

The external warning threshold in the locked policy is diagnostic for the
external forecast and is loaded into live RiskEngine. It does not replace
`pressure_critical_equivalent_depth_m=0.30`, which remains the separate
hydrostatic policy threshold.
