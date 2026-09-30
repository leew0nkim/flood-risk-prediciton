import json
import unittest
from collections import deque
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from meit_model.electronics_io import SensorPacket
from meit_model.run_scale_integration import main
from meit_model.scale_integration_model import ScaleIntegrationModel, StableScaleIntegrationModel


class ScaleIntegrationModelTests(unittest.TestCase):
    def setUp(self):
        self.model = ScaleIntegrationModel()

    def test_external_threshold_boundaries_with_zero_inside(self):
        cases = (
            (.0199, 'SAFE', 'NONE', False),
            (.02, 'WARNING', 'NONE', False),
            (.03, 'DANGER', 'OPEN', True),
        )
        for external, state, command, signal in cases:
            with self.subTest(external=external):
                result = self.model.evaluate(external, .0)
                self.assertEqual(result.state, state)
                self.assertEqual(result.external_state, state)
                self.assertEqual(result.internal_state, 'SAFE')
                self.assertEqual(result.command, command)
                self.assertEqual(result.open_signal, signal)
                self.assertAlmostEqual(result.depth_gap_m, external)

    def test_internal_limit_can_raise_state_but_cannot_open(self):
        result = self.model.evaluate(.01, .03)
        self.assertEqual(result.external_state, 'SAFE')
        self.assertEqual(result.internal_state, 'DANGER')
        self.assertEqual(result.state, 'DANGER')
        self.assertFalse(result.open_signal)
        self.assertEqual(result.command, 'CLOSE')

    def test_active_opening_does_not_repeat_command(self):
        result = self.model.evaluate(.03, .0, equalization_active=True)
        self.assertTrue(result.open_signal)
        self.assertEqual(result.command, 'NONE')

    def test_scale_depth_gap_controls_open_and_close(self):
        # A 3 cm outside / 0 cm inside scenario reaches the danger/open gap.
        opening = self.model.evaluate(.03, .0)
        self.assertAlmostEqual(opening.depth_gap_m, .03)
        self.assertEqual(opening.gap_state, 'DANGER')
        self.assertEqual(opening.command, 'OPEN')

        # The same outside level does not open at the 3 cm inside hard limit.
        equalized = self.model.evaluate(.03, .026)
        self.assertEqual(equalized.depth_gap_m, .004)
        self.assertEqual(equalized.gap_state, 'SAFE')
        self.assertEqual(equalized.command, 'NONE')

        # Once active, reaching the 3 cm inside limit closes the inlet.
        closing = self.model.evaluate(.03, .03, equalization_active=True)
        self.assertEqual(closing.internal_depth_m, .03)
        self.assertEqual(closing.command, 'CLOSE')
        stale_feedback = self.model.evaluate(.03, .03, equalization_active=False)
        self.assertEqual(stale_feedback.command, 'CLOSE')

    def test_raw_internal_limit_immediately_bypasses_median_filter(self):
        stable = StableScaleIntegrationModel()
        stable.update(.03, .0, equalization_active=True)
        stable.update(.03, .0, equalization_active=True)
        result = stable.update(.03, .03, equalization_active=True)
        self.assertEqual(result.internal_depth_m, .0)  # filtered diagnostic
        self.assertTrue(result.safety_override)
        self.assertEqual(result.internal_state, 'DANGER')
        self.assertEqual(result.command, 'CLOSE')

    def test_median_filter_rejects_one_splash_and_requires_three_samples(self):
        stable = StableScaleIntegrationModel()
        self.assertEqual(stable.update(.005, .0).command, 'NONE')
        self.assertEqual(stable.update(.04, .0).command, 'NONE')
        result = stable.update(.005, .0)
        self.assertTrue(result.filter_ready)
        self.assertEqual(result.command, 'NONE')
        self.assertEqual(result.external_depth_m, .005)

        stable = StableScaleIntegrationModel()
        self.assertEqual(stable.update(.03, .0).command, 'NONE')
        self.assertEqual(stable.update(.03, .0).command, 'NONE')
        result = stable.update(.03, .0)
        self.assertEqual(result.command, 'OPEN')

    def test_invalid_levels_and_thresholds_are_rejected(self):
        for value in (-.01, float('nan'), float('inf'), True, '0.05'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.model.evaluate(value, .02)
        with self.assertRaises(ValueError):
            ScaleIntegrationModel(warning_depth_m=.06, danger_depth_m=.05)
        with self.assertRaises(ValueError):
            ScaleIntegrationModel(internal_warning_depth_m=.03, internal_limit_depth_m=.03)
        with self.assertRaises(ValueError):
            ScaleIntegrationModel(warning_depth_gap_m=.03, open_depth_gap_m=.03)
        with self.assertRaises(ValueError):
            StableScaleIntegrationModel(window_samples=4)

    def test_dry_run_outputs_danger_and_open_at_three_cm(self):
        output = StringIO()
        with redirect_stdout(output):
            code = main(['--dry-run', '--external-depth-cm', '3', '--internal-depth-cm', '0'])
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload['state'], 'DANGER')
        self.assertEqual(payload['command'], 'OPEN')
        self.assertTrue(payload['mock'])
        self.assertEqual(set(payload), {'state', 'mock', 'command'})

    def test_serial_path_sends_one_compact_open_response(self):
        class FakeTransport:
            def __init__(self):
                self.packets = deque([SensorPacket(.03, .0, False, False)] * 3)
                self.last_line = None
                self.last_error = None
                self.sent = []
                self.closed = False

            def read_sensor_packet(self):
                self.last_line = '{valid sensor json}'
                return self.packets.popleft()

            def send_ai_result(self, payload):
                self.sent.append(payload)

            def close(self):
                self.closed = True

        transport = FakeTransport()
        with patch('meit_model.run_scale_integration.ElectronicsSerial', return_value=transport), \
                redirect_stdout(StringIO()):
            code = main(['--port', 'fake', '--max-messages', '3'])
        self.assertEqual(code, 0)
        self.assertEqual(transport.sent[:2], [
            {'state': 'DANGER', 'mock': False, 'command': 'NONE'},
            {'state': 'DANGER', 'mock': False, 'command': 'NONE'},
        ])
        self.assertEqual(transport.sent[2],
                         {'state': 'DANGER', 'mock': False, 'command': 'OPEN'})
        self.assertTrue(transport.closed)

    def test_observe_only_uses_real_state_but_blocks_actuation(self):
        class FakeTransport:
            def __init__(self):
                self.last_line = None
                self.last_error = None
                self.sent = []

            def read_sensor_packet(self):
                self.last_line = '{real sensor}'
                return SensorPacket(.03, .0, False, False)

            def send_ai_result(self, payload):
                self.sent.append(payload)

            def close(self):
                pass

        transport = FakeTransport()
        with patch('meit_model.run_scale_integration.ElectronicsSerial', return_value=transport), \
                redirect_stdout(StringIO()):
            code = main(['--port', 'fake', '--observe-only', '--max-messages', '3'])
        self.assertEqual(code, 0)
        self.assertEqual(transport.sent[-1],
                         {'state': 'DANGER', 'mock': False, 'command': 'NONE'})

    def test_full_dry_run_is_explicit_opt_in(self):
        output = StringIO()
        with redirect_stdout(output):
            code = main([
                '--dry-run', '--full-wire-payload',
                '--external-depth-cm', '7', '--internal-depth-cm', '2',
            ])
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload['open_signal'])
        self.assertEqual(payload['decision_basis'], 'CURRENT_DEPTH_ONLY')
        self.assertEqual(payload['prediction_source'], 'none')


if __name__ == '__main__':
    unittest.main()
