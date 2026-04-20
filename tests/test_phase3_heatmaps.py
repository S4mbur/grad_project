import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import server


class Phase3HeatmapTests(unittest.TestCase):
    def test_ensemble_attention_views_expose_consensus_disagreement_and_shared(self):
        attn_list = [
            np.array([0.9, 0.1, 0.2, 0.0], dtype=np.float32),
            np.array([0.8, 0.1, 0.3, 0.0], dtype=np.float32),
            np.array([0.85, 0.15, 0.25, 0.0], dtype=np.float32),
        ]
        views = server._build_ensemble_attention_views(attn_list)

        self.assertEqual(set(views.keys()), {"consensus", "disagreement", "shared"})
        self.assertEqual(int(np.argmax(views["consensus"])), 0)
        self.assertEqual(int(np.argmax(views["shared"])), 0)
        self.assertAlmostEqual(float(views["disagreement"][0]), 0.0, places=3)

    def test_disagreement_rises_when_models_diverge(self):
        attn_list = [
            np.array([0.9, 0.1, 0.0], dtype=np.float32),
            np.array([0.1, 0.9, 0.0], dtype=np.float32),
            np.array([0.9, 0.1, 0.0], dtype=np.float32),
        ]
        views = server._build_ensemble_attention_views(attn_list)
        self.assertGreater(float(views["disagreement"][1]), 0.5)

    def test_default_heatmap_views_switch_between_single_and_ensemble(self):
        single = server._default_heatmap_views(False)
        ensemble = server._default_heatmap_views(True)

        self.assertEqual(single[0]["key"], "attention")
        self.assertEqual([v["key"] for v in ensemble], ["consensus", "disagreement", "shared"])

    def test_contrastive_attention_views_support_melanoma_comparisons(self):
        tile_attention = np.array([0.9, 0.6, 0.2], dtype=np.float32)
        tile_scores = np.array([
            [0.1, 0.2, 0.3, 1.8],  # melanoma dominant
            [0.1, 0.2, 1.5, 0.4],  # scc dominant
            [0.1, 1.4, 0.2, 0.3],  # bcc dominant
        ], dtype=np.float32)

        views = server._build_contrastive_attention_views(tile_attention, tile_scores)

        self.assertIn("contrast_melanoma_vs_scc", views)
        self.assertIn("contrast_melanoma_vs_bcc", views)
        self.assertEqual(int(np.argmax(views["contrast_melanoma_vs_scc"])), 0)
        self.assertEqual(int(np.argmax(views["contrast_melanoma_vs_bcc"])), 0)


if __name__ == "__main__":
    unittest.main()
