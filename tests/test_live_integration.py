import unittest
from unittest.mock import Mock

from meit_model.integration_test_modes import IntegrationSession
from meit_model.electronics_io import SensorPacket
from meit_model.predictor import ForecastResult
from test_io import config


def packet(out=.2, inside=0., active=False):
    return SensorPacket(out, inside, False, active)


class LiveIntegrationTests(unittest.TestCase):
    def predictor(self):
        predictor = Mock()
        predictor.forecast_warning_depth_m = .29
        return predictor

    def test_jitter_checkpoint_inference_once_per_completed_bin(self):
        predictor = self.predictor()
        predictor.predict.return_value = ForecastResult('READY', [.2] * 15, None)
        session = IntegrationSession('live', config(), 0, predictor=predictor)
        session.update(packet(.1), 0)
        for second in range(1, 1801):
            result = session.update(packet(second / 10000), second + .001)
        self.assertEqual(result['prediction_source'], 'lstm')
        self.assertEqual(result['forecast_warning_depth_m'], .29)
        self.assertEqual(result['forecast_warning_policy_status'], 'LOCKED_FROM_CHECKPOINT')
        self.assertEqual(result['forecast_status'], 'READY')
        predictor.predict.assert_called_once()
        values = predictor.predict.call_args.args[0]
        self.assertEqual(len(values), 30)
        self.assertAlmostEqual(values[-1], .1799, places=6)
        self.assertEqual(predictor.predict.call_args.kwargs['origin_time'].timestamp(), 1800)
        updated = session.update(packet(.2, .35), 1802)
        self.assertEqual(updated['state'], 'DANGER')
        self.assertEqual(predictor.predict.call_count, 1)
        stale = session.update(None, 1983)
        self.assertEqual(stale['sensor_status'], 'STALE')
        self.assertEqual(stale['state'], 'DANGER')
        self.assertIsNone(stale['predicted_depths_m'])
        self.assertIsNone(stale['t_critical_min'])
        recovered = session.update(packet(), 1984)
        self.assertNotEqual(recovered['forecast_status'], 'READY')
        self.assertEqual(recovered['command'], 'NONE')

    def test_inference_error_preserves_current_flood_and_is_cached(self):
        predictor = self.predictor()
        predictor.predict.return_value = ForecastResult('PREDICTION_ERROR', None, None)
        session = IntegrationSession('live', config(), 0, predictor=predictor)
        for second in range(0, 1801, 60):
            result = session.update(packet(.5, .5), second)
        self.assertEqual(result['forecast_status'], 'PREDICTION_ERROR')
        self.assertEqual(result['state'], 'DANGER')
        session.update(packet(), 1801)
        predictor.predict.assert_called_once()

    def test_no_sensor_never_calls_model(self):
        predictor = self.predictor()
        result = IntegrationSession('live', config(), 0, predictor=predictor).update(None, 2000)
        predictor.predict.assert_not_called()
        self.assertEqual(result['state'], 'WARNING')
        self.assertIsNone(result['t_critical_min'])

    def test_live_mode_rejects_unlocked_cv_predictor(self):
        predictor = Mock()
        predictor.forecast_warning_depth_m = None
        with self.assertRaisesRegex(ValueError, 'locked warning policy'):
            IntegrationSession('live', config(), 0, predictor=predictor)
