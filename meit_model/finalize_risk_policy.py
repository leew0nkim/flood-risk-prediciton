"""Lock a human-reviewed threshold using window and 5-minute event reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from meit_model.fold_scaling import data_fingerprint
from meit_model.risk_policy import create_policy_decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--window-calibration", type=Path, required=True)
    parser.add_argument("--event-evaluation", type=Path, required=True)
    parser.add_argument("--selected-threshold", type=float, required=True)
    parser.add_argument("--selection-note", required=True,
                        help="Human review rationale comparing event recall, FPR, precision and alarm rate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing policy decision")
    try:
        decision = create_policy_decision(
            args.window_calibration, args.event_evaluation,
            data_fingerprint(args.data_dir), args.selected_threshold, args.selection_note)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")
    print(f"Locked warning threshold: {decision['forecast_warning_depth_m']:.3f} m")
    print(f"Policy decision: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
