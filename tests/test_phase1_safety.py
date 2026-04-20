import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import server


class Phase1SafetyTests(unittest.TestCase):
    def test_abstain_for_borderline_melanoma_case(self):
        probs = np.array([0.31, 0.34, 0.08, 0.27], dtype=np.float32)
        safety = server._build_phase1_safety(1, probs, ensemble_predictions=[1, 1, 3])
        self.assertTrue(safety['melanoma_first_guard'])
        self.assertTrue(safety['abstain_recommended'])
        self.assertEqual(safety['decision_status'], 'abstain')
        self.assertEqual(safety['display_prediction'], 'Needs Expert Review')
        self.assertEqual(safety['prediction_key'], 'abstain')

    def test_high_risk_melanoma_without_abstain(self):
        probs = np.array([0.03, 0.05, 0.04, 0.88], dtype=np.float32)
        safety = server._build_phase1_safety(3, probs)
        self.assertFalse(safety['abstain_recommended'])
        self.assertEqual(safety['decision_status'], 'predicted')
        self.assertEqual(safety['display_prediction'], 'Melanoma')
        self.assertEqual(safety['risk_level'], 'high risk')

    def test_low_risk_confident_non_melanoma_case(self):
        probs = np.array([0.06, 0.86, 0.05, 0.03], dtype=np.float32)
        safety = server._build_phase1_safety(1, probs)
        self.assertFalse(safety['melanoma_first_guard'])
        self.assertFalse(safety['abstain_recommended'])
        self.assertEqual(safety['risk_level'], 'low risk')
        self.assertEqual(safety['display_prediction'], 'BCC')

    def test_ensemble_disagreement_is_reported(self):
        probs = np.array([0.15, 0.44, 0.14, 0.27], dtype=np.float32)
        safety = server._build_phase1_safety(1, probs, ensemble_predictions=[1, 2, 3])
        self.assertIsNotNone(safety['ensemble_disagreement'])
        self.assertGreaterEqual(safety['ensemble_disagreement'], 0.33)


if __name__ == '__main__':
    unittest.main()
