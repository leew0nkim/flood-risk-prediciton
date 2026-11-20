# Canonical model release and final-evaluation gate

The canonical development release is `floodnet-391a996-canonical-v1`.
`configs/canonical_data_release.json` records the source commit, source CSV
hash, every processed artifact hash and the four-file development fingerprint.
Run the following before CV, final training or evaluation:

```bash
python -m meit_model.final_evaluation_gate
```

The existing `test_risk` and `test_temporal` splits were opened on 2026-09-29.
Their results are exploratory evidence only. They must not be used to choose a
model, epoch, loss, input feature or warning threshold, and they are not valid
as a new objective final holdout.

The completed development-only recipe is:

- four fixed chronological CV folds;
- seeds 42, 43 and 44, for 12 trained fold models;
- final epoch 14, the rounded median of the 12 best validation epochs;
- actual external crossing criterion 0.300m;
- forecast warning threshold 0.300m;
- development 5-minute event recall 17/19 for every threshold candidate;
- 0.300m selected because it had the highest precision and lowest FPR and
  alarm rate among candidates with equal event recall;
- final model trained on all 121,913 development windows without test access.

The final checkpoint is `artifacts/final_model/final_model.pt`. Its training
evidence and hashes are recorded in `analysis/a_plan_training_summary.json`.

Objective final evaluation remains blocked. A new holdout must be collected or
registered before inference. Its event IDs must be absent from train,
validation, test_risk and test_temporal; its selection rule and SHA-256 must be
frozen before the model is opened on it. Register it as `SEALED_UNOPENED` in the
release contract, then run:

```bash
python -m meit_model.final_evaluation_gate --holdout PATH_TO_NEW_HOLDOUT.npz
```

The source currently has no storm IDs. Event-level disjointness is enforced,
but storm-level independence must not be claimed. When `storm_id` becomes
available, every sensor and event from one storm must remain in one split.
