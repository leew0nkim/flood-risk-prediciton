import unittest
from itertools import product

from meit_model.controller import SafetyController, OPEN, CLOSE, NONE
from meit_model.risk import RiskResult


def risk(state):
    return RiskResult(state, .2, .29, .3, None, None, 'READY', 'test')


class ControllerTests(unittest.TestCase):
    def test_all_boolean_combinations_independent_of_led(self):
        controller = SafetyController()
        for evacuated, required, allowed, active in product((False, True), repeat=4):
            should_be_open = not evacuated and required and allowed
            expected = (OPEN if should_be_open else NONE) if not active else (
                NONE if should_be_open else CLOSE)
            for state in ('SAFE', 'WARNING', 'DANGER'):
                with self.subTest(inputs=(evacuated, required, allowed, active), state=state):
                    result = controller.decide(
                        risk(state), evacuation_complete=evacuated,
                        equalization_required=required, equalization_allowed=allowed,
                        equalization_active=active)
                    self.assertEqual(result.command, expected)
                    self.assertEqual(result.state, state)

    def test_invalid_values_rejected_before_any_early_return(self):
        controller = SafetyController()
        fields = ('evacuation_complete', 'equalization_required',
                  'equalization_allowed', 'equalization_active')
        invalid_values = (None, 'false', 'true', '', 0, 1, 0.0, 1.0, [], {}, float('nan'))
        for values in product((False, True), repeat=4):
            valid = dict(zip(fields, values))
            for field in fields:
                for invalid in invalid_values:
                    with self.subTest(valid=valid, field=field, invalid=invalid):
                        with self.assertRaisesRegex(TypeError, f'{field} must be bool'):
                            controller.decide(risk('DANGER'), **{**valid, field: invalid})


if __name__ == '__main__':
    unittest.main()
