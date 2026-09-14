import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from meit_data.io import (DEPTH, DRAIN_TIME, DURATION, END, Event, PEAK,
                          PEAK_TIME, SECONDS, SENSOR, START, assign_splits, read_events)


def config():
    return json.loads(Path("configs/pipeline_config.json").read_text())


class EventAuditTests(unittest.TestCase):
    def read_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            return read_events(path, config())

    def row(self, **overrides):
        row = {SENSOR: "sensor", START: "01/01/2024 12:00:00 AM", END: "01/01/2024 01:00:00 AM",
               DEPTH: "[0, 1, 1, 2]", SECONDS: "[0, 60, 60, 120]", PEAK: "2",
               PEAK_TIME: "1,500", DRAIN_TIME: "", DURATION: "60"}
        row.update(overrides)
        return row

    def test_exact_duplicate_removed_summary_error_does_not_impute_depth(self):
        events, audit, _ = self.read_rows([self.row()])
        self.assertEqual(audit.iloc[0].exact_duplicate_points_removed, 1)
        np.testing.assert_array_equal(events[0].seconds, [0, 60, 120])
        np.testing.assert_allclose(events[0].depth_m, [0, .0254, .0508])
        self.assertIn("peak_time_exceeds_duration", events[0].flags)
        self.assertEqual(events[0].quarantine, [])

    def test_conflicting_duplicate_and_bad_arrays_rejected(self):
        for change in ({DEPTH: "[0, 1, 9, 2]"}, {SECONDS: "[0, 60]"}, {DEPTH: "[0, -1, -1, 2]"}):
            with self.subTest(change=change):
                events, audit, _ = self.read_rows([self.row(**change)])
                self.assertEqual(events, [])
                self.assertEqual(audit.iloc[0].status, "invalid")

    def test_large_summary_difference_quarantined_without_clipping(self):
        events, audit, _ = self.read_rows([self.row(**{PEAK: "0.1"})])
        self.assertEqual(audit.iloc[0].status, "quarantined")
        self.assertAlmostEqual(events[0].depth_m.max(), .0508)

    def test_duplicate_record_and_ambiguous_key(self):
        events, audit, _ = self.read_rows([self.row(), self.row(), self.row(**{PEAK_TIME: "2"})])
        self.assertEqual(len(events), 2)
        self.assertEqual((audit.status == "duplicate_record").sum(), 1)
        self.assertTrue(all("duplicate_event_key" in e.quarantine for e in events))

    def test_split_uses_whole_dates_and_excludes_crossing_events(self):
        events = []
        for day in range(1, 31):
            for sensor in ("a", "b"):
                start = pd.Timestamp(f"2024-01-{day:02}T00:00:00Z")
                events.append(Event(f"{sensor}_{day}", sensor, start, start + pd.Timedelta(hours=1),
                                    day, np.array([0., 60.]), np.array([0., .1])))
        manifest = assign_splits(events, config())
        first_cut = pd.Timestamp(manifest["cut_dates_utc"][0])
        prior = next(e for e in events if e.start == first_cut - pd.Timedelta(days=1))
        prior.end = first_cut + pd.Timedelta(minutes=10)
        assign_splits(events, config())
        self.assertEqual(prior.split_reason, "boundary_crossing_event")
        self.assertTrue(all(e.split_reason == "embargo" for e in events if e.start == first_cut))
        by_split = {s: [e for e in events if e.split == s] for s in ("train", "validation", "test")}
        self.assertLess(max(e.effective_end for e in by_split["train"]), min(e.start for e in by_split["validation"]))
        self.assertLess(max(e.effective_end for e in by_split["validation"]), min(e.start for e in by_split["test"]))
        for date in {e.start for e in events}:
            self.assertLessEqual(len({e.split for e in events if e.start == date and e.split != "excluded"}), 1)


if __name__ == "__main__":
    unittest.main()
