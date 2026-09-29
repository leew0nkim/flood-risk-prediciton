import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta
import torch
from meit_model.predictor import ForecastPredictor, LiveForecastBuffer, ForecastResult
from meit_model.lstm import LSTMForecaster

class FakePredictor:
    def predict(self, values, *, origin_time):
        return ForecastResult('READY', [values[-1]] * 15, [origin_time] * 15)

class PredictorTests(unittest.TestCase):
    def test_time_grid_gap_missing_and_recovery(self):
        live = LiveForecastBuffer(FakePredictor()); t = datetime(2026, 1, 1)
        for i in range(35):
            result = live.add_measurement(.2, timestamp=t + timedelta(minutes=i))
            self.assertEqual(result.status, 'WARMUP' if i < 29 else 'READY')
        self.assertEqual(live.add_measurement(.2, timestamp=t + timedelta(minutes=34)).status, 'INSUFFICIENT_DATA')
        self.assertEqual(live.size,30)
        self.assertEqual(live.add_measurement(.2, timestamp=t + timedelta(minutes=36)).status,'WARMUP')
        self.assertEqual(live.size,1)
        self.assertEqual(live.add_measurement(None, timestamp=t + timedelta(minutes=37)).status,'INSUFFICIENT_DATA')
        self.assertFalse(live.ready)
        self.assertEqual(live.size,0)
        self.assertEqual(live.add_measurement(.2, timestamp=t + timedelta(minutes=38)).status,'WARMUP')
        self.assertEqual(live.size,1)

    def test_high_frequency_input_cannot_fill_history(self):
        live=LiveForecastBuffer(FakePredictor());t=datetime(2026,1,1)
        for i in range(35): live.add_measurement(.2,timestamp=t+timedelta(seconds=i))
        self.assertEqual(live.size,1)
        self.assertFalse(live.ready)
        live.add_measurement(.2,timestamp=t-timedelta(minutes=1))
        self.assertEqual(live.timestamps[-1],t)

    def test_checkpoint_config_scaler_and_invalid_output(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt'
            model=LSTMForecaster(hidden_size=8,num_layers=2)
            for parameter in model.parameters(): parameter.data.zero_()
            model.output_layer.bias.data.fill_(2)
            fingerprint={'train.npz':'current'}
            policy={'status':'LOCKED_FOR_FINAL_TRAINING',
                    'data_fingerprint':fingerprint,
                    'forecast_warning_depth_m':.29,
                    'external_critical_depth_m':.30,
                    'pressure_critical_equivalent_depth_m':.30,
                    'required_lead_time_min':5}
            torch.save({'config':{'hidden_size':8,'num_layers':2,'input_size':1,'horizon':15},
                        'scaler':{'mean_m':.1,'std_m':.2},'model_state_dict':model.state_dict(),
                        'data_fingerprint':fingerprint,'policy':policy},path)
            predictor=ForecastPredictor(path,device='cpu')
            self.assertEqual(predictor.forecast_warning_depth_m,.29)
            result=predictor.predict([.2]*30)
            self.assertEqual(result.status,'READY')
            self.assertTrue(all(abs(x-.5)<1e-6 for x in result.predicted_depths_m))
            predictor.model.output_layer.bias.data.fill_(float('nan'))
            result=predictor.predict([.2]*30)
            self.assertEqual(result.status,'PREDICTION_ERROR')
            self.assertIsNone(result.predicted_depths_m)
            self.assertEqual(predictor.predict([None]*30).status,'INSUFFICIENT_DATA')
            self.assertEqual(predictor.predict([{}]*30).status,'INSUFFICIENT_DATA')

if __name__=='__main__': unittest.main()
