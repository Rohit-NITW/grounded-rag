import json
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from grounded.config import Settings
from grounded.models import Source
from grounded.providers import Provider, ProviderError


class ProviderTests(unittest.TestCase):
    def make_provider(self, handler, provider="ollama"):
        settings = Settings(_env_file=None, provider=provider, openai_api_key="test-only")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        instance = Provider(settings, client)
        self.addCleanup(instance.close)
        return instance

    def test_ollama_embedding_contract_and_prefixes(self):
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            self.assertEqual(request.url.path, "/api/embed")
            return httpx.Response(200, json={"embeddings": [[3, 4]]})
        provider = self.make_provider(handle)
        self.assertTrue(np.allclose(provider.embed(["hello"]), [[0.6, 0.8]]))
        provider.embed(["hello"], query=True)
        self.assertEqual(requests[0]["input"], ["search_document: hello"])
        self.assertEqual(requests[1]["input"], ["search_query: hello"])
        self.assertFalse(requests[0]["truncate"])

    def test_openai_reorders_embedding_results(self):
        def handle(request):
            self.assertEqual(request.url.path, "/v1/embeddings")
            self.assertEqual(request.headers["authorization"], "Bearer test-only")
            return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 1]},
                                                     {"index": 0, "embedding": [1, 0]}]})
        self.assertTrue(np.array_equal(self.make_provider(handle, "openai").embed(["a", "b"]), [[1, 0], [0, 1]]))

    def test_invalid_embedding_count_and_values(self):
        for values in [[], [[0, 0]], [["bad", 1]], [[1, 2], [2, 3]]]:
            with self.subTest(values=values):
                provider = self.make_provider(lambda request: httpx.Response(200, json={"embeddings": values}))
                with self.assertRaises(ProviderError):
                    provider.embed(["hello"])

    def test_transient_embedding_error_retries(self):
        calls = []
        def handle(request):
            calls.append(request)
            return httpx.Response(429 if len(calls) == 1 else 200, json={"embeddings": [[1, 0]]})
        with patch("grounded.providers.time.sleep"):
            self.make_provider(handle).embed(["hello"])
        self.assertEqual(len(calls), 2)

    def test_upstream_secrets_not_exposed(self):
        provider = self.make_provider(lambda request: httpx.Response(401, json={"error": "secret private content"}))
        with self.assertRaises(ProviderError) as raised:
            provider.embed(["hello"])
        self.assertNotIn("secret private content", str(raised.exception))

    def test_generation_contracts_and_usage(self):
        sources = [Source(source_id="S1", chunk_id="chunk", document_id="doc", filename="x.md",
                          page=1, text="Backups last thirty days.", score=1, cosine=1, bm25=1)]
        output = {"insufficient_evidence": False,
                  "claims": [{"text": "Thirty days.", "source_ids": ["S1"]}]}
        for name in ["ollama", "openai"]:
            def handle(request):
                data = json.loads(request.content)
                if name == "ollama":
                    self.assertFalse(data["stream"])
                    self.assertIn("properties", data["format"])
                    self.assertIn("untrusted", data["messages"][0]["content"])
                    return httpx.Response(200, json={"message": {"content": json.dumps(output)},
                                                     "prompt_eval_count": 50, "eval_count": 12})
                self.assertEqual(request.url.path, "/v1/responses")
                self.assertFalse(data["store"])
                self.assertTrue(data["text"]["format"]["strict"])
                return httpx.Response(200, json={"output": [{"type": "message", "content": [
                    {"type": "output_text", "text": json.dumps(output)}]}],
                    "usage": {"input_tokens": 50, "output_tokens": 12}})
            with self.subTest(provider=name):
                answer, usage = self.make_provider(handle, name).generate("retention?", sources)
                self.assertEqual(answer.claims[0].source_ids, ["S1"])
                self.assertEqual(usage["input_tokens"], 50)

    def test_malformed_generation_fails_cleanly(self):
        provider = self.make_provider(lambda request: httpx.Response(200, json={"message": {"content": "not json"}}))
        with self.assertRaisesRegex(ProviderError, "invalid structured"):
            provider.generate("hello", [])

    def test_openai_incomplete_output_fails_cleanly(self):
        provider = self.make_provider(lambda request: httpx.Response(200, json={"status": "incomplete"}), "openai")
        with self.assertRaisesRegex(ProviderError, "truncated"):
            provider.generate("hello", [])


if __name__ == "__main__":
    unittest.main()
