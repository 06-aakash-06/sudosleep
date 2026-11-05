import io
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from siegen_dataset import archive_records, load_segment, recording_info
from wearable_pipeline import _find_axis_columns, infer_sample_rate, heuristic_scores, heuristic_shap_attributions


class SiegenDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.archive = Path(self.tmp.name) / "study.zip"
        dtype = [("t", "<f8"), ("d", "u1"), ("x", "u1"), ("y", "u1"),
                 ("z", "u1"), ("l", "<u2"), ("gt", "<i8")]
        self.runs = np.zeros(61, dtype=dtype)
        self.runs["t"] = 734633.0 + np.arange(61) / 86400.0
        self.runs["d"] = 100
        self.runs["x"] = np.arange(61) + 90
        self.runs["y"] = 120
        self.runs["z"] = 150
        self.runs["gt"] = np.where(np.arange(61) < 30, 6, 2)
        buf = io.BytesIO()
        np.save(buf, self.runs, allow_pickle=False)
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("data/p001.npy", buf.getvalue())
            archive.writestr("data/pat_inf.npy", b"not a recording")

    def tearDown(self):
        self.tmp.cleanup()

    def test_expansion_and_stages_are_not_apnea_labels(self):
        self.assertEqual(archive_records(self.archive), ["p001"])
        frame = load_segment(self.archive, "p001", 0, 60)
        self.assertGreaterEqual(len(frame), 5998)
        self.assertAlmostEqual(float(np.median(np.diff(frame.time_s))), 0.01, places=6)
        self.assertEqual(int(frame.loc[frame.time_s < 29, "sleep_stage_code"].mode()[0]), 6)
        self.assertEqual(int(frame.loc[frame.time_s > 31, "sleep_stage_code"].mode()[0]), 2)
        self.assertNotIn("label", frame.columns)
        self.assertAlmostEqual(infer_sample_rate(frame)[0], 100.0, places=3)
        self.assertGreater(recording_info(self.archive, "p001")["duration_seconds"], 59)
        self.assertAlmostEqual(recording_info(self.archive, "p001")["first_sleep_seconds"], 30, places=1)

    def test_invalid_record_and_out_of_range_are_rejected(self):
        with self.assertRaises(ValueError):
            load_segment(self.archive, "../p001", 0, 60)
        with self.assertRaises(ValueError):
            load_segment(self.archive, "p001", 1000, 60)

    def test_old_competition_accel_column_names_are_supported(self):
        import pandas as pd
        columns = _find_axis_columns(pd.DataFrame(columns=["accel_x", "accel_y", "accel_z"]))
        self.assertEqual(columns, {"x": "accel_x", "y": "accel_y", "z": "accel_z"})

    def test_sleep_stage_codes_are_rejected_as_apnea_training_labels(self):
        source = Path(self.tmp.name) / "stages.csv"
        source.write_text("record,subject,x,y,z,label\nr1,s1,0,0,1,2\n")
        result = subprocess.run([sys.executable, "train_wearable.py", str(source)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PSG sleep stages are not apnea labels", result.stderr)

    def test_heuristic_and_shap_use_each_record_baseline(self):
        scalars = np.zeros((6, 12), dtype=np.float32)
        scalars[:, 2] = [1, 2, 3, 100, 200, 300]
        scalars[:, 3] = [1, 2, 3, 100, 200, 300]
        data = {"scalars": scalars, "pause_fraction": np.zeros(6),
                "records": np.array(["a", "a", "a", "b", "b", "b"])}
        scores, _ = heuristic_scores(data)
        np.testing.assert_allclose(scores[:3], scores[3:], atol=1e-6)
        _, values, expected, predicted = heuristic_shap_attributions(data, 3)
        self.assertAlmostEqual(predicted, float(scores[3]), places=5)
        self.assertAlmostEqual(expected + values.sum(), predicted, places=5)


if __name__ == "__main__":
    unittest.main()
