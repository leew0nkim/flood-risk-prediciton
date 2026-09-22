import unittest
from meit_model.predictor import ForecastResult
from meit_model.risk import RiskEngine


class RiskEngineTests(unittest.TestCase):
    def test_led_policy_and_external_reference(self):
        engine = RiskEngine()
        cases = [
            (.20, [.20] * 15, 'SAFE', None, None),
            (.26, [.26] * 15, 'WARNING', None, None),
            (.22, [.27] * 9 + [.291] * 6, 'WARNING', 10, None),
            (.22, [.27, .275, .280, .291] + [.295] * 11, 'DANGER', 4, None),
            (.22, [.27] * 4 + [.291] * 11, 'DANGER', 5, None),
            (.22, [.27] * 5 + [.291] * 10, 'WARNING', 6, None),
            (.22, [.27] * 3 + [.291] * 6 + [.31] * 6, 'DANGER', 4, 10),
            (.305, [.31] * 15, 'DANGER', 0, 0),
        ]
        for current, values, state, warning, critical in cases:
            with self.subTest(current=current, values=values):
                result = engine.evaluate(current, ForecastResult('READY', values, None))
                self.assertEqual((result.state, result.t_warning_min,
                                  result.t_external_critical_min), (state, warning, critical))
                self.assertFalse(hasattr(result, 'command'))
                self.assertFalse(hasattr(result, 't_critical_min'))

    def test_unavailable_forecast_and_current_danger_priority(self):
        engine = RiskEngine()
        for status in ['WARMUP', 'INSUFFICIENT_DATA', 'PREDICTION_ERROR']:
            forecast = ForecastResult(status, None, None)
            self.assertEqual(engine.evaluate(.20, forecast).state, 'WARNING')
            self.assertEqual(engine.evaluate(.30, forecast).state, 'DANGER')
        for values in [[], [.2] * 14, [float('nan')] * 15]:
            result = engine.evaluate(.2, ForecastResult('READY', values, None))
            self.assertEqual(result.state, 'WARNING')
            self.assertEqual(result.forecast_status, 'INVALID_FORECAST')

    def test_invalid_policy(self):
        for options in [{'forecast_warning_depth_m': float('nan')},
                        {'forecast_warning_depth_m': .31},
                        {'external_critical_depth_m': float('inf')},
                        {'danger_horizon_min': 5.5}]:
            with self.assertRaises(ValueError):
                RiskEngine(**options)
