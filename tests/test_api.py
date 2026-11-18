import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from grounded.api import create_app
from grounded.config import Settings
from grounded.providers import ProviderError


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        settings = Settings(_env_file=None, provider="demo", data_dir=Path(self.temp.name), api_token="test-token",
                            max_upload_bytes=2048)
        self.app = create_app(settings)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.headers = {"Authorization": "Bearer test-token"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def upload(self, content=b"Backups are retained for thirty days."):
        return self.client.post("/api/documents", files={"file": ("guide.md", content)}, headers=self.headers)

    def test_health_and_ui_public_api_protected(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn("frame-ancestors 'none'", home.headers["Content-Security-Policy"])
        self.assertEqual(self.client.get("/api/documents").status_code, 401)
        self.assertEqual(self.client.get("/api/config", headers={"Authorization": "Bearer wrong"}).status_code, 401)

    def test_upload_query_search_delete_roundtrip(self):
        response = self.upload()
        self.assertEqual(response.status_code, 201)
        identifier = response.json()["document_id"]
        self.assertEqual(self.upload().json()["duplicate"], True)
        listing = self.client.get("/api/documents", headers=self.headers).json()
        self.assertEqual(len(listing["documents"]), 1)
        answer = self.client.post("/api/ask", json={"question": "How long are backups retained?"}, headers=self.headers)
        self.assertEqual(answer.status_code, 200)
        self.assertIn("thirty days", answer.json()["answer"])
        self.assertEqual(answer.headers["X-Request-ID"], answer.json()["request_id"])
        search = self.client.post("/api/search", json={"question": "backups", "strategy": "bm25"}, headers=self.headers)
        self.assertEqual(len(search.json()["sources"]), 1)
        self.assertEqual(self.client.delete(f"/api/documents/{identifier}", headers=self.headers).status_code, 200)
        self.assertEqual(self.client.delete(f"/api/documents/{identifier}", headers=self.headers).status_code, 404)

    def test_validation(self):
        for value in [{"question": " "}, {"question": "ok", "top_k": 100},
                      {"question": "ok", "strategy": "invented"}, {"question": "ok", "extra": True}]:
            with self.subTest(value=value):
                self.assertEqual(self.client.post("/api/ask", json=value, headers=self.headers).status_code, 422)

    def test_unsupported_file(self):
        response = self.client.post("/api/documents", files={"file": ("x.exe", b"hello")}, headers=self.headers)
        self.assertEqual(response.status_code, 400)

    def test_upload_file_limit(self):
        self.assertEqual(self.upload(b"a" * 2049).status_code, 413)

    def test_whole_request_limit(self):
        result = self.client.post("/api/documents", content=b"a" * 70_000, headers=self.headers)
        self.assertEqual(result.status_code, 413)

    def test_provider_failure_is_actionable(self):
        with patch.object(self.app.state.service.provider, "embed", side_effect=ProviderError("Model unavailable")):
            result = self.upload()
        self.assertEqual(result.status_code, 502)
        self.assertEqual(result.json()["detail"], "Model unavailable")
        self.assertEqual(self.client.get("/api/documents", headers=self.headers).json()["documents"], [])

    def test_config_does_not_leak_secrets(self):
        response = self.client.get("/api/config", headers=self.headers)
        self.assertNotIn("test-token", response.text)
        self.assertNotIn("openai_api_key", response.text)


if __name__ == "__main__":
    unittest.main()
