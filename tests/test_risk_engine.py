import unittest
from meit_model.predictor import ForecastResult
from meit_model.risk import RiskEngine


class RiskEngineTests(unittest.TestCase):
    def test_led_policy_and_external_reference(self):
        engine = RiskEngine()
        cases = [
            (.20, [.20] * 15, 'SAFE', None, None),
            (.26, [.26] * 15, 'WARNING', None, None),
            (.22, [.27] * 9 + [.291] * 6, 'SAFE', None, None),
            (.22, [.27, .275, .280, .291] + [.295] * 11, 'SAFE', 5, None),
            (.22, [.27] * 4 + [.291] * 11, 'SAFE', None, None),
            (.22, [.27] * 5 + [.291] * 10, 'SAFE', None, None),
            (.22, [.27] * 3 + [.291] * 6 + [.31] * 6, 'WARNING', 10, 10),
            (.305, [.31] * 15, 'DANGER', 0, 0),
        ]
        for current, values, state, warning, critical in cases:
            with self.subTest(current=current, values=values):
                result = engine.evaluate(current, ForecastResult('READY', values, None), internal_depth_m=0., equalization_active=False)
                self.assertEqual((result.state, result.t_warning_min,
                                  result.t_external_critical_min), (state, warning, critical))
                self.assertFalse(hasattr(result, 'command'))
                self.assertEqual(result.t_external_warning_min, warning)

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
                        {'danger_horizon_min': 5.5},
                        {'internal_warning_depth_m': .31},
                        {'internal_critical_depth_m': float('nan')},
                        {'pressure_critical_equivalent_depth_m': 0},
                        {'pressure_critical_equivalent_depth_m': float('inf')}]:
            with self.assertRaises(ValueError):
                RiskEngine(**options)

    def evaluate_pair(self, outside, inside, values=None, **options):
        options.setdefault('equalization_active', False)
        return RiskEngine().evaluate(outside, ForecastResult('READY', values or [.2] * 15, None),
            internal_depth_m=inside, **options)

    def test_equal_gaps_do_not_have_equal_hydrostatic_loads(self):
        shallow = self.evaluate_pair(.2, 0.)
        deep = self.evaluate_pair(.5, .3)
        self.assertAlmostEqual(shallow.depth_gap_m, deep.depth_gap_m)
        self.assertAlmostEqual(shallow.bottom_pressure_difference_pa, 1961.33)
        self.assertAlmostEqual(shallow.bottom_pressure_difference_pa, deep.bottom_pressure_difference_pa)
        self.assertAlmostEqual(shallow.equivalent_depth_m, .2)
        self.assertAlmostEqual(deep.equivalent_depth_m, .4)
        # Integrated force per width is proportional to equivalent depth squared.
        self.assertAlmostEqual((deep.equivalent_depth_m / shallow.equivalent_depth_m) ** 2, 4.)

    def test_equalized_water_does_not_clear_flood_danger(self):
        result = self.evaluate_pair(.5, .5, [.5] * 15)
        self.assertEqual(result.equivalent_depth_m, 0.)
        self.assertEqual(result.bottom_pressure_difference_pa, 0.)
        self.assertEqual(result.pressure_direction, 'BALANCED')
        self.assertEqual(result.forecast_state, 'SAFE')
        self.assertEqual((result.external_state, result.internal_state, result.state), ('DANGER',) * 3)

    def test_analytical_pressure_boundaries_are_inclusive(self):
        self.assertEqual(self.evaluate_pair(.5, .4).t_critical_min, 0)
        self.assertEqual(self.evaluate_pair(.5, .3).t_critical_min, 0)

    def test_reverse_pressure_and_indoor_flood_are_retained(self):
        result = self.evaluate_pair(.3, .5)
        self.assertAlmostEqual(result.equivalent_depth_m, .4)
        self.assertAlmostEqual(result.depth_gap_m, -.2)
        self.assertLess(result.bottom_pressure_difference_pa, 0.)
        self.assertEqual(result.pressure_direction, 'INSIDE_HIGHER')
        self.assertEqual(result.state, 'DANGER')
        result = self.evaluate_pair(.2, .31)
        self.assertEqual(result.external_state, 'SAFE')
        self.assertEqual(result.internal_state, 'DANGER')
        self.assertEqual(result.state, 'DANGER')

    def test_missing_indoor_is_unknown_and_never_overrides_known_danger(self):
        for outside, state in [(.2, 'WARNING'), (.5, 'DANGER')]:
            result = self.evaluate_pair(outside, None)
            self.assertEqual(result.state, state)
            self.assertEqual(result.pressure_status, 'MISSING_INTERNAL_DEPTH')
            self.assertIsNone(result.depth_gap_m)
            self.assertIsNone(result.equivalent_depth_m)
            self.assertIsNone(result.t_pressure_critical_if_internal_constant_min)

    def test_invalid_indoor_and_external_depths_rejected(self):
        for bad in [-.01, float('nan'), float('inf'), True, '0.2']:
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.evaluate_pair(.2, bad)
            with self.subTest(external=bad), self.assertRaises(ValueError):
                self.evaluate_pair(bad, 0.)

    def test_conditional_pressure_time_uses_internal_depth(self):
        values = [.3] * 2 + [.5] * 13
        low = self.evaluate_pair(.2, .3, values, equalization_active=False)
        high = self.evaluate_pair(.2, .41, values, equalization_active=False)
        self.assertEqual(low.pressure_forecast_status, 'CONDITIONAL_INTERNAL_CONSTANT')
        self.assertEqual(low.t_pressure_critical_if_internal_constant_min, 3)
        self.assertIsNone(high.t_pressure_critical_if_internal_constant_min)
        # Original external crossing remains external, not a door-opening time.
        self.assertEqual(low.t_external_critical_min, 1)
        self.assertEqual(high.t_external_critical_min, 1)

    def test_pressure_time_unavailable_when_equalizing_or_unknown(self):
        for active in (True, None):
            result = self.evaluate_pair(.2, .1, [.5] * 15, equalization_active=active)
            self.assertTrue(result.pressure_forecast_status.startswith('UNAVAILABLE'))
            self.assertIsNone(result.t_pressure_critical_if_internal_constant_min)

    def test_large_finite_depths_never_emit_nonfinite_json(self):
        import json
        from dataclasses import asdict
        result = self.evaluate_pair(1e300, 5e299)
        self.assertAlmostEqual(result.equivalent_depth_m / 1e300, 3 ** .5 / 2)
        json.dumps(asdict(result), allow_nan=False)
        result = self.evaluate_pair(1e308, 0., equalization_active=False)
        self.assertEqual(result.pressure_status, 'NUMERIC_OUT_OF_RANGE')
        self.assertIsNone(result.bottom_pressure_difference_pa)
        self.assertEqual(result.pressure_forecast_status, 'UNAVAILABLE_NUMERIC_RANGE')
        self.assertIsNone(result.t_pressure_critical_if_internal_constant_min)
        json.dumps(asdict(result), allow_nan=False)

    def test_invalid_forecast_cannot_hide_known_indoor_danger(self):
        result = self.evaluate_pair(.1, .4, [float('nan')] * 15)
        self.assertEqual(result.forecast_status, 'INVALID_FORECAST')
        self.assertEqual(result.state, 'DANGER')
        self.assertIsNone(result.t_pressure_critical_if_internal_constant_min)


class FuturePressurePolicyTests(unittest.TestCase):
    def test_lead_boundaries_and_internal_level(self):
        for minute, expected in [(4, 'DANGER'), (5, 'DANGER'), (6, 'WARNING'), (15, 'WARNING')]:
            values = [.22] * (minute - 1) + [.30] * (16 - minute)
            result = RiskEngine().evaluate(.22, ForecastResult('READY', values, None),
                internal_depth_m=0., equalization_active=False)
            self.assertEqual(result.t_critical_min, minute)
            self.assertEqual(result.state, expected)
        result = RiskEngine().evaluate(.22, ForecastResult('READY', [.30] * 15, None),
            internal_depth_m=.20, equalization_active=False)
        self.assertEqual(result.t_external_critical_min, 1)
        self.assertIsNone(result.t_critical_min)
        self.assertEqual(result.state, 'SAFE')

    def test_reverse_pressure_is_not_future_opening_risk(self):
        result = RiskEngine().evaluate(.10, ForecastResult('READY', [.10] * 15, None),
            internal_depth_m=.40, equalization_active=False)
        self.assertIsNone(result.t_critical_min)
        self.assertEqual(result.forecast_state, 'SAFE')
        self.assertEqual(result.state, 'DANGER')


class CurrentCrossingTests(unittest.TestCase):
    def test_external_times_independently_include_current_depth(self):
        for status, values in [('READY', [.2] * 15), ('WARMUP', None)]:
            for depth, warning, critical in [(.289, None, None), (.29, None, None), (.295, 0, None), (.30, 0, 0)]:
                with self.subTest(status=status, depth=depth):
                    r = RiskEngine().evaluate(depth, ForecastResult(status, values, None))
                    self.assertEqual((r.t_external_warning_min, r.t_external_critical_min), (warning, critical))

    def test_current_pressure_reached_without_usable_forecast(self):
        for status, values in [('READY', [.2] * 15), ('READY', [float('nan')] * 15),
                               ('WARMUP', None), ('INSUFFICIENT_DATA', None), ('PREDICTION_ERROR', None)]:
            for active in (False, True, None):
                with self.subTest(status=status, active=active):
                    r = RiskEngine().evaluate(.5, ForecastResult(status, values, None),
                        internal_depth_m=.4, equalization_active=active)
                    self.assertEqual(r.t_critical_min, 0)
                    self.assertEqual(r.t_pressure_critical_if_internal_constant_min, 0)
                    self.assertEqual(r.pressure_forecast_status, 'CURRENT_CRITICAL_REACHED')
                    self.assertEqual((r.state, r.forecast_state), ('DANGER', 'DANGER'))
                    self.assertEqual(r.forecast_status, 'INVALID_FORECAST' if status == 'READY' and values[0] != values[0] else status)

    def test_no_false_current_crossing(self):
        for outside, inside in [(.4, .5), (.5, .5), (.2, 0.), (.5, None)]:
            r = RiskEngine().evaluate(outside, ForecastResult('WARMUP', None, None), internal_depth_m=inside)
            self.assertIsNone(r.t_critical_min)
