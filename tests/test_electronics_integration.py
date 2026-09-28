import json
import unittest
from collections import deque
from unittest.mock import patch

import numpy as np

from meit_model.electronics_io import ElectronicsSerial, SensorPacket
from meit_model.integration_test_modes import IntegrationSession
from meit_model.run_electronics_integration import main
from test_io import config


def packet(depth=.2, internal=.0, active=False):
    return SensorPacket(depth, internal, False, active)


class FakeSerial:
    def __init__(self, chunks=()):
        self.chunks = deque(chunks)
        self.written = []
        self.closed = False

    def readline(self, size):
        return self.chunks.popleft() if self.chunks else b''

    def write(self, value):
        self.written.append(value)
        return len(value)

    def close(self):
        self.closed = True


def wire(depth=.2, internal=.0):
    return (json.dumps(dict(external_depth_m=depth, internal_depth_m=internal,
                           evacuation_complete=False, equalization_active=False)) + '\n').encode()


class ElectronicsTests(unittest.TestCase):
    def test_quick_cycle_immediate_and_explicitly_mock(self):
        session = IntegrationSession('quick', config(), 1000)
        for elapsed, expected, crossing in [(0, 'SAFE', None), (5, 'WARNING', 10),
                                            (10, 'DANGER', 3), (15, 'SAFE', None)]:
            # Real telemetry must not override this deliberately synthetic LED test.
            result = session.update(packet(.5), 1000 + elapsed)
            self.assertEqual(result['state'], expected)
            self.assertEqual(result['forecast_status'], 'READY')
            self.assertEqual(result['history_status'], 'BYPASSED')
            self.assertEqual(result['prediction_source'], 'mock')
            self.assertEqual(result['received_sensor_depth_m'], .5)
            self.assertEqual(result['current_depth_m'], .2)
            self.assertEqual(result['t_external_critical_min'], crossing)
            self.assertEqual(result['t_pressure_critical_if_internal_constant_min'], crossing)
            self.assertEqual(result['internal_depth_m'], .1)
            self.assertEqual(result['received_internal_depth_m'], 0.)
            self.assertEqual(result['command'], 'NONE')
            self.assertEqual(len(result['predicted_depths_m']), 15)

    def test_real_history_with_jitter_uses_last_completed_minute(self):
        session = IntegrationSession('accumulate', config(), 0)
        session.update(packet(.1), 0)
        for i in range(1, 1801):
            result = session.update(packet(i / 10000), i + .001)
            if i < 1800:
                self.assertNotEqual(result['forecast_status'], 'READY')
        self.assertEqual(result['forecast_status'], 'READY')
        self.assertEqual(result['history_bins'], 30)
        self.assertEqual(result['observed_history_bins'], 30)
        expected = (np.arange(60, 1801, 60) - 1) / 10000
        np.testing.assert_allclose(session.input_result['X'][0, :, 0], expected)
        self.assertAlmostEqual(result['predicted_depths_m'][0], .1799, places=6)
        self.assertEqual(result['forecast_origin_unix_seconds'], 1800)

    def test_packet_count_cannot_replace_elapsed_history(self):
        session = IntegrationSession('accumulate', config(), 0)
        for i in range(3000):
            result = session.update(packet(), i / 100)
        self.assertEqual(result['history_bins'], 0)
        self.assertEqual(result['forecast_status'], 'WARMUP')
        self.assertIsNone(result['predicted_depths_m'])

    def test_stale_without_packets_and_long_gap_recovery(self):
        session = IntegrationSession('accumulate', config(), 0)
        for second in range(0, 1801, 60):
            result = session.update(packet(), second)
        self.assertEqual(result['forecast_status'], 'READY')
        result = session.update(None, 1981)
        self.assertEqual(result['sensor_status'], 'STALE')
        self.assertEqual(result['state'], 'WARNING')
        self.assertEqual(result['forecast_status'], 'INSUFFICIENT_DATA')
        self.assertIsNone(result['predicted_depths_m'])
        result = session.update(packet(), 2040)
        self.assertEqual(result['history_bins'], 0)
        for second in range(2100, 3841, 60):
            result = session.update(packet(), second)
        self.assertEqual(result['forecast_status'], 'READY')

    def test_missing_bin_uses_causal_fill_and_no_data_never_ready(self):
        session = IntegrationSession('accumulate', config(), 0)
        for second in range(0, 1801, 60):
            result = session.update(None if second == 900 else packet(), second)
        self.assertEqual(result['forecast_status'], 'READY')
        self.assertEqual(result['observed_history_bins'], 29)
        self.assertFalse(session.input_result['X_observed'][0, 14, 0])
        empty = IntegrationSession('accumulate', config(), 0).update(None, 1800)
        self.assertEqual(empty['forecast_status'], 'INSUFFICIENT_DATA')
        self.assertEqual(empty['sensor_status'], 'NO_DATA')

    def test_serial_fragments_invalid_values_and_roundtrip(self):
        value = wire()
        fake = FakeSerial([value[:15], b'', value[15:]])
        transport = ElectronicsSerial('fake', connection=fake)
        self.assertIsNone(transport.read_sensor_packet())
        self.assertIsNone(transport.read_sensor_packet())
        self.assertEqual(transport.read_sensor_packet(), packet())
        for bad in (True, 'NaN', float('nan'), float('inf'), -.1):
            fake.chunks.append(wire(bad))
            self.assertIsNone(transport.read_sensor_packet())
            self.assertIsNotNone(transport.last_error)
        fake.chunks.append(wire())
        received = transport.read_sensor_packet()
        result = IntegrationSession('quick', config(), 0, 'danger').update(received, 0)
        transport.send_ai_result(result)
        self.assertEqual(json.loads(fake.written[0])['state'], 'DANGER')
        self.assertTrue(fake.written[0].endswith(b'\n'))
        self.assertIsNone(transport.last_error)

    def test_cli_runs_without_checkpoint_and_closes_transport(self):
        fake = FakeSerial([wire()])
        transport = ElectronicsSerial('fake', connection=fake)
        with patch('meit_model.run_electronics_integration.ElectronicsSerial', return_value=transport), patch('builtins.print'):
            code = main(['--mode', 'quick', '--scenario', 'warning', '--port', 'fake', '--max-messages', '1'])
        self.assertEqual(code, 0)
        self.assertTrue(fake.closed)
        self.assertEqual(json.loads(fake.written[0])['state'], 'WARNING')
        self.assertEqual(json.loads(fake.written[0]),
                         {'state': 'WARNING', 'mock': True, 'command': 'NONE'})

    def test_cli_exposes_board_messages_as_status_not_parse_errors(self):
        fake = FakeSerial([b'Arduino Ready\n'])
        transport = ElectronicsSerial('fake', connection=fake)
        with patch('meit_model.run_electronics_integration.ElectronicsSerial', return_value=transport), \
                patch('builtins.print') as printed:
            code = main(['--mode', 'quick', '--scenario', 'safe', '--port', 'fake',
                         '--max-messages', '1'])
        self.assertEqual(code, 0)
        payload = json.loads(printed.call_args_list[-1].args[0])
        self.assertEqual(payload['serial_input_event'], {'type': 'BOARD_READY'})
        self.assertTrue(payload['board_ready'])
        self.assertEqual(payload['board_sensor_status'], 'UNKNOWN')
        self.assertIsNone(payload['serial_input_error'])
        self.assertEqual(json.loads(fake.written[0]),
                         {'state': 'SAFE', 'mock': True, 'command': 'NONE'})

    def test_reply_on_input_logs_real_values_and_only_replies_to_valid_lines(self):
        first, second = wire(.42, .12), wire(.17, .08)
        fake = FakeSerial([b'', b'not-json\n', first[:20], b'', first[20:], second])
        transport = ElectronicsSerial('fake', connection=fake)
        with patch('meit_model.run_electronics_integration.ElectronicsSerial', return_value=transport), \
                patch('meit_model.run_electronics_integration.time.sleep'), \
                patch('meit_model.predictor.ForecastPredictor', side_effect=AssertionError('No model in serial mock test')), \
                patch('builtins.print') as printed:
            code = main(['--mode', 'quick', '--scenario', 'safe', '--reply-on-input',
                         '--port', 'fake', '--max-messages', '2'])
        self.assertEqual(code, 0)
        self.assertTrue(fake.closed)
        self.assertFalse(fake.chunks)  # No response before a complete, valid packet.
        self.assertEqual(len(fake.written), 2)
        for response in fake.written:
            self.assertLess(len(response), 256)
            self.assertTrue(response.endswith(b'\n'))
            payload = json.loads(response)
            self.assertEqual(payload['state'], 'SAFE')
            self.assertIs(payload['mock'], True)
            self.assertEqual(payload['command'], 'NONE')
        logs = [str(call.args[0]) for call in printed.call_args_list]
        self.assertEqual(sum(line.startswith('[RX OK]') for line in logs), 2)
        self.assertTrue(any('[RX OK]' in line and '0.42' in line and '0.12' in line for line in logs))
        self.assertTrue(any('[RX ERROR]' in line and 'not-json' in line for line in logs))
        self.assertEqual(sum(line.startswith('[TX MOCK]') for line in logs), 2)

    def test_full_wire_payload_is_explicit_opt_in(self):
        fake = FakeSerial([wire()])
        transport = ElectronicsSerial('fake', connection=fake)
        with patch('meit_model.run_electronics_integration.ElectronicsSerial', return_value=transport), \
                patch('builtins.print'):
            code = main(['--mode', 'quick', '--scenario', 'safe', '--port', 'fake',
                         '--full-wire-payload', '--max-messages', '1'])
        self.assertEqual(code, 0)
        payload = json.loads(fake.written[0])
        self.assertEqual(payload['prediction_source'], 'mock')
        self.assertEqual(payload['predicted_depths_m'], [.2] * 15)

    def test_reply_on_input_rejects_modes_without_live_sensor_requests(self):
        for options in (['--mode', 'quick', '--dry-run'], ['--mode', 'accumulate', '--port', 'fake']):
            with self.subTest(options=options), patch('sys.stderr'), self.assertRaises(SystemExit) as caught:
                main(options + ['--reply-on-input'])
            self.assertEqual(caught.exception.code, 2)

    def test_minute_interval_jitter_and_oversized_line_recovery(self):
        session = IntegrationSession('accumulate', config(), 0)
        session.update(packet(), 0)
        for minute in range(1, 31):
            result = session.update(packet(), minute * 60 + .001)
        self.assertEqual(result['forecast_status'], 'READY')
        fake = FakeSerial([b'x' * 4097, b'end\n', wire()])
        transport = ElectronicsSerial('fake', connection=fake)
        self.assertIsNone(transport.read_sensor_packet())
        self.assertIsNone(transport.read_sensor_packet())
        self.assertEqual(transport.read_sensor_packet(), packet())
        with patch.object(fake, 'write', return_value=0):
            with self.assertRaisesRegex(OSError, 'Incomplete serial write'):
                transport.send_ai_result(result)

    def test_indoor_warning_during_warmup_and_stale_danger_retention(self):
        session = IntegrationSession('accumulate', config(), 0)
        result = session.update(packet(.1, .35), 0)
        self.assertEqual(result['forecast_status'], 'WARMUP')
        self.assertEqual(result['internal_state'], 'DANGER')
        self.assertEqual(result['state'], 'DANGER')
        self.assertEqual(result['pressure_direction'], 'INSIDE_HIGHER')
        stale = session.update(None, 181)
        self.assertEqual(stale['sensor_status'], 'STALE')
        self.assertEqual(stale['pressure_status'], 'STALE')
        self.assertEqual(stale['state'], 'DANGER')
        self.assertIsNone(stale['equivalent_depth_m'])
        self.assertIsNone(stale['t_pressure_critical_if_internal_constant_min'])
        fresh = session.update(packet(.1, .0), 182)
        self.assertEqual(fresh['state'], 'WARNING')  # A new, valid warmup pair.

    def test_same_packet_indoor_depth_is_used_after_history_ready(self):
        session = IntegrationSession('accumulate', config(), 0)
        for second in range(0, 1801, 60):
            result = session.update(packet(.5, .3), second)
        self.assertEqual(result['forecast_status'], 'READY')
        self.assertAlmostEqual(result['equivalent_depth_m'], .4)
        self.assertEqual(result['t_pressure_critical_if_internal_constant_min'], 0)
        equalized = session.update(packet(.5, .5), 1801)
        self.assertEqual(equalized['equivalent_depth_m'], 0.)
        self.assertEqual(equalized['pressure_state'], 'SAFE')
        self.assertEqual(equalized['state'], 'DANGER')
        active = session.update(packet(.5, .3, True), 1802)
        self.assertEqual(active['pressure_forecast_status'], 'UNAVAILABLE_EQUALIZATION_ACTIVE')
        self.assertIsNone(active['t_pressure_critical_if_internal_constant_min'])

    def test_invalid_internal_packet_does_not_refresh_pair(self):
        fake = FakeSerial([wire(.2, .1), wire(.6, None)])
        transport = ElectronicsSerial('fake', connection=fake)
        session = IntegrationSession('accumulate', config(), 0)
        session.update(transport.read_sensor_packet(), 0)
        result = session.update(transport.read_sensor_packet(), 181)
        self.assertIsNotNone(transport.last_error)
        self.assertEqual(result['received_sensor_depth_m'], .2)
        self.assertEqual(result['received_internal_depth_m'], .1)
        self.assertEqual(result['sensor_status'], 'STALE')
        with self.assertRaises(ValueError):
            session.update(packet(.5, None), 182)

    def test_equalized_and_inside_high_quick_scenarios(self):
        for scenario in ('equalized', 'inside-high'):
            result = IntegrationSession('quick', config(), 0, scenario).update(packet(.0, .0), 0)
            self.assertEqual(result['state'], 'DANGER')
            self.assertEqual(result['prediction_source'], 'mock')
            self.assertEqual(result['command'], 'NONE')
            if scenario == 'equalized':
                self.assertEqual(result['equivalent_depth_m'], 0.)
            else:
                self.assertEqual(result['pressure_direction'], 'INSIDE_HIGHER')


if __name__ == '__main__':
    unittest.main()
