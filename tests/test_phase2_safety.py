import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import server


class Phase2SafetyTests(unittest.TestCase):
    def setUp(self):
        server._phase2_registry_cache.clear()

    def test_probability_calibration_uses_temperature_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "calibration_registry.json"
            reg_path.write_text(json.dumps({
                "phikon_cost_sensitive_strong": {
                    "method": "temperature_scaling",
                    "temperature": 2.0,
                    "ece_before": 0.11,
                    "ece_after": 0.05,
                }
            }), encoding="utf-8")

            with mock.patch.object(server, "PHASE2_CALIBRATION_PATH", reg_path):
                calibrated, meta = server._apply_probability_calibration(
                    np.array([0.90, 0.05, 0.03, 0.02], dtype=np.float32),
                    "phikon_cost_sensitive_strong",
                )

            self.assertTrue(meta["available"])
            self.assertAlmostEqual(meta["temperature"], 2.0, places=3)
            self.assertLess(calibrated.max(), 0.90)
            self.assertAlmostEqual(float(calibrated.sum()), 1.0, places=5)

    def test_phase2_merges_ood_signal_without_forcing_abstain_when_low(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ood_path = Path(tmpdir) / "ood_registry.json"
            ood_path.write_text(json.dumps({
                "phikon_cost_sensitive_strong": {
                    "class_centroids": {
                        "BCC": [0.0, 0.0, 0.0],
                        "Melanoma": [5.0, 5.0, 5.0],
                    },
                    "class_thresholds": {
                        "BCC": 2.0,
                        "Melanoma": 2.0,
                    }
                }
            }), encoding="utf-8")

            phase1 = server._build_phase1_safety(1, np.array([0.08, 0.83, 0.06, 0.03], dtype=np.float32))
            with mock.patch.object(server, "PHASE2_OOD_PATH", ood_path):
                merged = server._merge_phase2_safety(
                    phase1,
                    np.array([0.08, 0.83, 0.06, 0.03], dtype=np.float32),
                    "phikon_cost_sensitive_strong",
                    np.array([0.3, 0.2, 0.1], dtype=np.float32),
                    {"temperature": 1.3, "ece_after": 0.04},
                )

            self.assertEqual(merged["decision_status"], "predicted")
            self.assertEqual(merged["prediction_key"], "bcc")
            self.assertTrue(merged["ood"]["available"])
            self.assertLess(merged["ood"]["ood_score"], 0.2)
            self.assertIn("calibration", merged)
            self.assertIn("safety_score", merged)

    def test_phase2_strong_ood_forces_abstain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ood_path = Path(tmpdir) / "ood_registry.json"
            ood_path.write_text(json.dumps({
                "phikon_cost_sensitive_strong": {
                    "class_centroids": {
                        "BCC": [0.0, 0.0, 0.0],
                        "Melanoma": [1.0, 1.0, 1.0],
                    },
                    "class_thresholds": {
                        "BCC": 0.5,
                        "Melanoma": 0.5,
                    }
                }
            }), encoding="utf-8")

            phase1 = server._build_phase1_safety(1, np.array([0.10, 0.72, 0.08, 0.10], dtype=np.float32))
            with mock.patch.object(server, "PHASE2_OOD_PATH", ood_path):
                merged = server._merge_phase2_safety(
                    phase1,
                    np.array([0.10, 0.72, 0.08, 0.10], dtype=np.float32),
                    "phikon_cost_sensitive_strong",
                    np.array([4.0, 4.0, 4.0], dtype=np.float32),
                    {"temperature": 1.0},
                )

            self.assertTrue(merged["abstain_recommended"])
            self.assertEqual(merged["decision_status"], "abstain")
            self.assertEqual(merged["display_prediction"], "Needs Expert Review")
            self.assertTrue(merged["ood"]["ood_flag"])


if __name__ == "__main__":
    unittest.main()
