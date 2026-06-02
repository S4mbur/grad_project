import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import server  # noqa: E402


class AppRoutesSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_models_endpoint_exposes_gated_default(self):
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["default"], server.DEFAULT_MODEL_KEY)
        gated = [item for item in data["ensembles"] if item["key"] == server.DEFAULT_MODEL_KEY]
        self.assertEqual(len(gated), 1)
        self.assertTrue(gated[0]["gated"])
        self.assertEqual(gated[0]["gating_policy"]["name"], "cheap_conf70_margin20_mel20")

    def test_info_endpoint_returns_registry_summary(self):
        response = self.client.get("/api/info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("n_models", data)
        self.assertIn("retrieval_banks", data)


if __name__ == "__main__":
    unittest.main()
