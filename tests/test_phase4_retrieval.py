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


class Phase4RetrievalTests(unittest.TestCase):
    def setUp(self):
        server._phase4_registry_cache.clear()
        server.analyses.clear()

    def test_single_model_retrieval_returns_best_match_and_hard_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            registry_path = tmp / "retrieval_registry.json"
            embeddings_path = tmp / "retrieval_embeddings.npz"

            registry_path.write_text(json.dumps({
                "cases": {
                    "case_a": {
                        "slide_id": "case_a",
                        "filename": "case_a.tif",
                        "true_label": "BCC",
                        "source": "cobra_bcc",
                        "thumbnail_url": "/api/retrieval/thumbnails/case_a.jpg",
                        "is_hard_melanoma": False,
                    },
                    "case_b": {
                        "slide_id": "case_b",
                        "filename": "case_b.tif",
                        "true_label": "Melanoma",
                        "source": "tcga_skcm",
                        "thumbnail_url": "/api/retrieval/thumbnails/case_b.jpg",
                        "is_hard_melanoma": True,
                    },
                },
                "banks": {
                    "phikon_cost_sensitive_strong": {
                        "display": "Phikon - Cost-Sensitive Strong",
                        "type": "single_model",
                        "case_ids": ["case_a", "case_b"],
                        "n_cases": 2,
                        "hard_case_count": 1,
                    }
                },
            }), encoding="utf-8")
            np.savez_compressed(
                embeddings_path,
                phikon_cost_sensitive_strong=np.asarray([
                    [1.0, 0.0],
                    [0.0, 1.0],
                ], dtype=np.float32),
            )

            with mock.patch.object(server, "PHASE4_RETRIEVAL_PATH", registry_path), \
                 mock.patch.object(server, "PHASE4_EMBEDDINGS_PATH", embeddings_path):
                result = server._retrieve_similar_cases(
                    "phikon_cost_sensitive_strong",
                    bag_embedding=np.asarray([0.98, 0.02], dtype=np.float32),
                    top_k=2,
                    hard_top_k=1,
                )

            self.assertTrue(result["available"])
            self.assertEqual(result["similar_cases"][0]["slide_id"], "case_a")
            self.assertEqual(result["hard_melanoma_matches"][0]["slide_id"], "case_b")

    def test_ensemble_retrieval_uses_concatenated_component_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            registry_path = tmp / "retrieval_registry.json"
            embeddings_path = tmp / "retrieval_embeddings.npz"

            registry_path.write_text(json.dumps({
                "cases": {
                    "case_a": {
                        "slide_id": "case_a",
                        "filename": "case_a.tif",
                        "true_label": "Melanoma",
                        "source": "tcga_skcm",
                        "thumbnail_url": "/api/retrieval/thumbnails/case_a.jpg",
                        "is_hard_melanoma": True,
                    },
                    "case_b": {
                        "slide_id": "case_b",
                        "filename": "case_b.tif",
                        "true_label": "BCC",
                        "source": "cobra_bcc",
                        "thumbnail_url": "/api/retrieval/thumbnails/case_b.jpg",
                        "is_hard_melanoma": False,
                    },
                },
                "banks": {
                    "ensemble_3_best": {
                        "display": "Ensemble 3-Model (UNI + Phikon + CONCH)",
                        "type": "ensemble",
                        "case_ids": ["case_a", "case_b"],
                        "n_cases": 2,
                        "hard_case_count": 1,
                    }
                },
            }), encoding="utf-8")
            np.savez_compressed(
                embeddings_path,
                ensemble_3_best=np.asarray([
                    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                ], dtype=np.float32) / np.sqrt(3.0),
            )

            with mock.patch.object(server, "PHASE4_RETRIEVAL_PATH", registry_path), \
                 mock.patch.object(server, "PHASE4_EMBEDDINGS_PATH", embeddings_path):
                result = server._retrieve_similar_cases(
                    "ensemble_3_best",
                    ensemble_model_keys=["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong", "conch_cost_sensitive_strong"],
                    ensemble_bag_embeddings=[
                        np.asarray([1.0, 0.0], dtype=np.float32),
                        np.asarray([1.0, 0.0], dtype=np.float32),
                        np.asarray([1.0, 0.0], dtype=np.float32),
                    ],
                    top_k=1,
                    hard_top_k=1,
                )

            self.assertTrue(result["available"])
            self.assertEqual(result["bank_type"], "ensemble")
            self.assertEqual(result["similar_cases"][0]["slide_id"], "case_a")

    def test_compare_route_returns_cached_result_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            registry_path = tmp / "retrieval_registry.json"
            embeddings_path = tmp / "retrieval_embeddings.npz"

            registry_path.write_text(json.dumps({
                "cases": {
                    "case_a": {
                        "slide_id": "case_a",
                        "filename": "case_a.tif",
                        "slide_path": "/tmp/case_a.tif",
                        "true_label": "Melanoma",
                        "source": "tcga_skcm",
                        "thumbnail_url": "/api/retrieval/thumbnails/case_a.jpg",
                        "is_hard_melanoma": True,
                    },
                },
                "banks": {},
            }), encoding="utf-8")
            np.savez_compressed(embeddings_path, dummy=np.asarray([[1.0]], dtype=np.float32))

            job_id = server._comparison_job_id("case_a", "ensemble_3_best")
            server.analyses[job_id] = {
                "status": "completed",
                "filename": "case_a.tif",
                "slide_path": "/tmp/case_a.tif",
                "model_key": "ensemble_3_best",
                "model_display": server.ENSEMBLE_PRESETS["ensemble_3_best"]["display"],
                "created_at": "2026-04-05T22:00:00",
                "result": {
                    "prediction": "Melanoma",
                    "raw_prediction": "Melanoma",
                    "decision_status": "predicted",
                    "safety": {"risk_level": "high risk"},
                    "top_tiles": [],
                    "heatmap_views": [{"key": "consensus", "label": "Consensus"}],
                    "default_heatmap_view": "consensus",
                    "artifacts": {"export_url": f"/api/results/{job_id}/export"},
                },
            }

            with mock.patch.object(server, "PHASE4_RETRIEVAL_PATH", registry_path), \
                 mock.patch.object(server, "PHASE4_EMBEDDINGS_PATH", embeddings_path):
                client = server.app.test_client()
                resp = client.get("/api/retrieval/cases/case_a/compare?model=ensemble_3_best")

            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload["slide_id"], "case_a")
            self.assertEqual(payload["result"]["prediction"], "Melanoma")
            self.assertIn("artifacts", payload["result"])

    def test_export_route_includes_policy_artifacts_and_retrieval(self):
        server.analyses["job123"] = {
            "status": "completed",
            "filename": "sample.tif",
            "slide_path": "/tmp/sample.tif",
            "model_key": "ensemble_3_best",
            "model_display": server.ENSEMBLE_PRESETS["ensemble_3_best"]["display"],
            "created_at": "2026-04-05T22:00:00",
            "slide_info": {"width": 1000, "height": 800},
            "result": {
                "prediction": "Needs Expert Review",
                "raw_prediction": "Melanoma",
                "decision_status": "abstain",
                "safety": {"risk_level": "urgent review recommended"},
                "retrieval": {"available": True, "similar_cases": [{"slide_id": "x"}]},
                "top_tiles": [],
                "heatmap_views": [{"key": "consensus", "label": "Consensus"}],
                "default_heatmap_view": "consensus",
            },
        }

        client = server.app.test_client()
        resp = client.get("/api/results/job123/export")
        self.assertEqual(resp.status_code, 200)
        payload = json.loads(resp.data.decode("utf-8"))
        resp.close()
        self.assertIn("decision_policy", payload)
        self.assertIn("artifacts", payload)
        self.assertIn("retrieval_summary", payload)


if __name__ == "__main__":
    unittest.main()
