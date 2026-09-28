"""Deterministic policy checks, not empirical LSTM accuracy or door validation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from meit_model.predictor import ForecastResult
from meit_model.risk import RiskEngine


def evaluate_scenarios():
    # name, outside, inside, forecast, status, active, expected LED, expected time
    cases = [
        ('safe', .2, 0., [.2]*15, 'READY', False, 'SAFE', None),
        ('outside_warning', .26, 0., [.26]*15, 'READY', False, 'WARNING', None),
        ('inside_warning', .2, .26, [.2]*15, 'READY', False, 'WARNING', None),
        ('five_minute_boundary', .22, 0., [.22]*4+[.30]*11, 'READY', False, 'DANGER', 5),
        ('six_minute_boundary', .22, 0., [.22]*5+[.30]*10, 'READY', False, 'WARNING', 6),
        ('ten_minute_warning', .22, 0., [.22]*9+[.30]*6, 'READY', False, 'WARNING', 10),
        ('indoor_reduces_pressure', .22, .2, [.30]*15, 'READY', False, 'SAFE', None),
        ('external_warning_diagnostic_only', .2, .2, [.295]*15, 'READY', False, 'SAFE', None),
        ('equalized_flood', .5, .5, [.5]*15, 'READY', False, 'DANGER', None),
        ('reverse_pressure', .1, .4, [.1]*15, 'READY', False, 'DANGER', None),
        ('already_critical_during_warmup', .5, .4, None, 'WARMUP', True, 'DANGER', 0),
        ('missing_inside', .2, None, [.5]*15, 'READY', False, 'WARNING', None),
        ('equalization_active', .2, .1, [.5]*15, 'READY', True, 'WARNING', None),
        ('equalization_unknown', .2, .1, [.5]*15, 'READY', None, 'WARNING', None),
        ('invalid_forecast', .2, 0., [float('nan')]*15, 'READY', False, 'WARNING', None),
    ]
    rows = []
    for name, outside, inside, values, status, active, state, time in cases:
        risk = RiskEngine().evaluate(outside, ForecastResult(status, values, None),
            internal_depth_m=inside, equalization_active=active)
        rows.append(dict(scenario=name, expected_state=state, expected_t_critical_min=time,
                         passed=risk.state == state and risk.t_critical_min == time,
                         result=asdict(risk)))
    return dict(scope='Synthetic policy verification; not model performance or hardware validation',
                count=len(rows), passed=sum(r['passed'] for r in rows), scenarios=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/led_policy/scenarios.json'))
    args = parser.parse_args()
    report = evaluate_scenarios()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"LED scenarios: {report['passed']}/{report['count']} passed; {args.output}")
    return 0 if report['passed'] == report['count'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
