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

    def test_gap_boundaries_ignore_absolute_depths_and_active_feedback(self):
        for internal in (0.0, .03, .50):
            for active in (False, True):
                for gap, expected in ((.0199, 'SAFE'), (.02, 'WARNING'), (.03, 'WARNING'),
                                      (.0349, 'WARNING'), (.035, 'DANGER'), (.04, 'DANGER')):
                    with self.subTest(internal=internal, active=active, gap=gap):
                        result = self.model.evaluate(internal + gap, internal,
                                                     equalization_active=active)
                        self.assertEqual(result.state, expected)
                        self.assertEqual(result.led, {'SAFE': 'BLUE', 'WARNING': 'GREEN', 'DANGER': 'RED'}[expected])
                        self.assertEqual(result.open_signal, gap >= .035)
                        if gap >= .035:
                            self.assertEqual(result.command, 'OPEN')

    def test_equal_or_higher_inside_is_safe(self):
        for external, internal in ((.5, .5), (.1, .2)):
            result = self.model.evaluate(external, internal)
            self.assertEqual(result.state, 'SAFE')
            self.assertEqual(result.command, 'NONE')

    def test_live_requires_three_samples_before_opening(self):
        stable = StableScaleIntegrationModel()
        for _ in range(2):
            result = stable.update(.07, .03)
            self.assertFalse(result.filter_ready)
            self.assertEqual(result.command, 'NONE')
        result = stable.update(.07, .03)
        self.assertTrue(result.filter_ready)
        self.assertEqual(result.command, 'OPEN')

    def test_five_sample_median_rejects_spike_and_tracks_sustained_change(self):
        stable = StableScaleIntegrationModel()
        for _ in range(5):
            stable.update(.04, .03)
        spike = stable.update(.10, .0)
        self.assertAlmostEqual(spike.external_depth_m, .04)
        self.assertAlmostEqual(spike.internal_depth_m, .03)
        self.assertEqual(spike.state, 'SAFE')
        self.assertEqual(spike.command, 'NONE')
        self.assertEqual(spike.filter_sample_count, 5)
        self.assertEqual(stable.update(.07, .03).command, 'NONE')
        result = stable.update(.07, .03)
        self.assertEqual(result.state, 'DANGER')
        self.assertEqual(result.command, 'OPEN')

    def test_three_sample_indicator_reports_warning_before_five_sample_filter(self):
        stable = StableScaleIntegrationModel()
        for _ in range(5):
            stable.update(.01, 0)
        self.assertEqual(stable.update(.03, 0).state, 'SAFE')
        fast_warning = stable.update(.03, 0)
        self.assertEqual(fast_warning.state, 'WARNING')
        self.assertEqual(fast_warning.led, 'GREEN')
        self.assertEqual(fast_warning.command, 'NONE')
        self.assertAlmostEqual(fast_warning.depth_gap_m, .03)
        self.assertAlmostEqual(fast_warning.actuation_depth_gap_m, .01)

    def test_existing_external_low_close_rule_is_preserved(self):
        self.assertEqual(self.model.evaluate(.02, .0, equalization_active=True).command, 'CLOSE')

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
        with self.assertRaises(ValueError):
            StableScaleIntegrationModel(indicator_window_samples=2)
        with self.assertRaises(ValueError):
            StableScaleIntegrationModel(window_samples=3, indicator_window_samples=5)

    def test_dry_run_outputs_danger_and_open_at_three_point_five_cm(self):
        output = StringIO()
        with redirect_stdout(output):
            code = main(['--dry-run', '--external-depth-cm', '3.5', '--internal-depth-cm', '0'])
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload['state'], 'DANGER')
        self.assertEqual(payload['command'], 'OPEN')
        self.assertTrue(payload['mock'])
        self.assertEqual(set(payload), {'state', 'led', 'mock', 'command'})

    def test_serial_path_sends_one_compact_open_response(self):
        class FakeTransport:
            def __init__(self):
                self.packets = deque([SensorPacket(.04, .0, False, False)] * 3)
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
            {'state': 'DANGER', 'led': 'RED', 'mock': False, 'command': 'NONE'},
            {'state': 'DANGER', 'led': 'RED', 'mock': False, 'command': 'NONE'},
        ])
        self.assertEqual(transport.sent[2],
                         {'state': 'DANGER', 'led': 'RED', 'mock': False, 'command': 'OPEN'})
        self.assertTrue(transport.closed)

    def test_observe_only_uses_real_state_but_blocks_actuation(self):
        class FakeTransport:
            def __init__(self):
                self.last_line = None
                self.last_error = None
                self.sent = []

            def read_sensor_packet(self):
                self.last_line = '{real sensor}'
                return SensorPacket(.04, .0, False, False)

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
                         {'state': 'DANGER', 'led': 'RED', 'mock': False, 'command': 'NONE'})

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
        self.assertEqual(payload['decision_basis'], 'CURRENT_DEPTH_GAP')
        self.assertEqual(payload['prediction_source'], 'none')


if __name__ == '__main__':
    unittest.main()
