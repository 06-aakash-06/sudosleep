import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from audit_dreamt import audit_file, EVENTS
from prepare_dreamt import derive_epoch_targets
from train_dreamt import event_match_counts, intervals_from_mask, smooth_by_night


class DreamtResearchTest(unittest.TestCase):
    def test_targets_require_scored_sleep_and_one_second_event_overlap(self):
        times = np.arange(120 * 64) / 64.0
        stages = np.full(len(times), "N2", dtype="U7")
        stages[:10 * 64] = "W"
        stages[60 * 64:] = "Missing"
        events = np.zeros(len(times), dtype=bool)
        events[20 * 64:21 * 64] = True
        labels, sleep = derive_epoch_targets(times, stages, events)
        np.testing.assert_array_equal(labels, [1, -1])
        self.assertAlmostEqual(float(sleep[0]), 50 / 60, places=5)

    def test_audit_counts_event_run_across_chunk_boundary_once(self):
        count = 200_010
        data = {"TIMESTAMP": np.arange(count) / 64.0, "Sleep_Stage": "N2",
                "ACC_X": 0, "ACC_Y": 0, "ACC_Z": 1}
        for name in EVENTS:
            values = np.full(count, np.nan)
            if name == "Hypopnea":
                values[199_998:200_003] = 1
            data[name] = values
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "S001_whole_df.csv"
            pd.DataFrame(data).to_csv(path, index=False)
            result = audit_file(path)
        self.assertEqual(result["event_runs_all_stages"], 1)
        self.assertEqual(result["positive_rows"], 5)
        self.assertEqual(result["gaps"], 0)

    def test_event_matching_is_one_to_one(self):
        reference = [(10, 30), (40, 50), (180, 200)]
        predicted = intervals_from_mask([True, False, False, True])
        self.assertEqual(event_match_counts(reference, predicted), (2, 0, 1))

    def test_smoothing_does_not_cross_unscored_windows(self):
        class FakeDataset:
            nights = [{"labels": np.array([0, -1, 0])}]
        scores = np.array([0.9, 0.9])
        transition = np.array([[0.9, 0.1], [0.1, 0.9]])
        from wearable_pipeline import viterbi_smooth
        expected = np.array([viterbi_smooth(scores[:1], transition)[0],
                             viterbi_smooth(scores[1:], transition)[0]])
        np.testing.assert_array_equal(smooth_by_night(FakeDataset(), scores, transition), expected)

    def test_conditional_model_shap_reconstructs_prediction(self):
        import torch
        from wearable_pipeline import FEATURE_NAMES, explain_accelerometer_model_scalars

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.01))

            def forward(self, signal, scalars):
                return scalars[:, 2, :].sum(dim=1, keepdim=True) * self.weight

        data = {"signals_3ch": np.zeros((1, 3, 6000), dtype=np.float32),
                "scalars": np.ones((1, 12), dtype=np.float32), "records": np.array(["s"]),
                "preprocessing_version": "emd_scg_v2_rate_aware"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save({"input_modality": "accelerometer", "preprocessing_version": data["preprocessing_version"],
                        "feature_names": FEATURE_NAMES, "model_states": [FakeModel().state_dict()],
                        "scalar_background": [0.0] * 12, "scalar_center": [0.0] * 12,
                        "scalar_scale": [2.0] * 12}, path)
            with patch("pipeline.SudoSleepNet", FakeModel), patch("pipeline.get_device", return_value=torch.device("cpu")):
                _, values, expected, predicted = explain_accelerometer_model_scalars(data, path, 0)
        self.assertAlmostEqual(expected + values.sum(), predicted, places=5)
        self.assertAlmostEqual(predicted, 1 / (1 + np.exp(-0.06)), places=5)


if __name__ == "__main__":
    unittest.main()
