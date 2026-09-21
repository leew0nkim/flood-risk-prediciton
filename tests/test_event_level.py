import unittest
import numpy as np
from meit_model.evaluate_event_level import evaluate_events, summarize_events

class EventEvaluationTests(unittest.TestCase):
    def test_group_sort_mask_and_first_alarm(self):
        ids=np.array([1,1,1,2,3]); times=np.array([300,0,60,0,0])
        current=np.array([.2,.2,.2,.2,.31])
        target=np.zeros((5,15)); mask=np.ones((5,15),bool); pred=np.zeros((5,15))
        # The same crossing at t=600 appears in three overlapping windows.
        target[0,4:]=.31; target[1,9:]=.31; target[2,8:]=.31
        target[1,0]=.99; mask[1,0]=False
        pred[0,0]=.29; pred[2,0]=.29
        target[3,4]=.31; target[4,:]=.4
        rows=evaluate_events(ids,times,current,target,mask,pred,.29)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['actual_crossing_time'],600)
        self.assertEqual(rows[0]['first_alarm_time'],60)
        self.assertEqual(rows[0]['lead_time_min'],9.)
        self.assertFalse(rows[1]['detected'])
        self.assertEqual(summarize_events(rows)['event_timely_recall'],.5)

    def test_alarm_outside_lookback_excluded_and_empty_summary(self):
        target=np.zeros((2,15)); target[1,14]=.31
        pred=np.zeros((2,15)); pred[0,:]=.4
        rows=evaluate_events(np.array([1,1]),np.array([0,1200]),np.array([.1,.1]),
                             target,np.ones((2,15),bool),pred,.29)
        self.assertFalse(rows[0]['detected'])
        self.assertIsNone(summarize_events([])['event_timely_recall'])

    def test_operational_boundary_and_missed_denominator(self):
        rows = [{'detected': True, 'lead_time_min': 5.},
                {'detected': True, 'lead_time_min': 1.},
                {'detected': False, 'lead_time_min': None}]
        summary = summarize_events(rows, 5.)
        self.assertEqual(summary['operational_timely_event_count'], 1)
        self.assertEqual(summary['operational_event_recall'], 1 / 3)
        self.assertEqual(summary['event_recall'], 2 / 3)
        self.assertIsNone(summarize_events(rows)['operational_event_recall'])
        self.assertIsNone(summarize_events([], 5)['operational_event_recall'])
        for invalid in [-1, 16, float('nan')]:
            with self.assertRaises(ValueError):
                summarize_events(rows, invalid)

    def test_invalid_predictions_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_events(np.array([1]),np.array([0]),np.array([.1]),np.zeros((1,15)),
                            np.ones((1,15),bool),np.full((1,15),np.nan),.29)

if __name__=='__main__': unittest.main()
