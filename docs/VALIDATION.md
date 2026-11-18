# Delivery validation record

Prepared November 18, 2026 (Asia/Kolkata).

## Executed successfully

- Python 3.12.14: `python -m unittest discover -s tests -v` — **42 tests passed**.
- `python -m grounded ingest samples` — six documents, twelve passages created.
- `python -m grounded.evaluate --generate --output reports/demo-benchmark.json` — all 24 cases evaluated under BM25, hash-vector, and hybrid strategies.
- API integration tests used FastAPI TestClient and exercised real upload → index → search → answer → delete behavior against temporary SQLite databases.
- PDF tests extracted a generated text PDF and rejected blank/scanned and encrypted PDFs.
- Ollama and OpenAI request/response contract tests used `httpx.MockTransport`; no remote API was contacted by those tests.
- Python compilation checks and JavaScript syntax check completed.

The checked-in benchmark is the actual demo-mode report from this environment. Its generation step is extractive, not an LLM. Timing values are specific to this small run.

## Included but not verified end to end here

- **Live Ollama and OpenAI inference:** no local model runtime or user API credentials were available. Run `python scripts/live_smoke.py` after configuring your provider.
- **Docker build and Compose startup:** Docker was not available in the execution environment.
- **Browser integration script:** Chromium could not launch because the execution sandbox denied a required socket operation. The separate browser environment did not reach this app's localhost server. The UI script and CI job are included, but are not reported as passed.
- **GitHub Actions:** workflow files were created, but no remote repository was created or CI run triggered.
- **Python 3.11 and 3.13:** included in CI configuration; only Python 3.12 was executed here.

## Before using it in an interview

1. Install dependencies in a clean virtual environment.
2. Run the core tests, browser smoke test, and live-provider smoke test.
3. Try several positive and negative questions, and inspect actual supporting passages.
4. Run your own held-out evaluation. Preserve its configuration and raw output.
5. If demonstrating Docker or CI, run those workflows and record their real outcomes.

No production scale, hallucination rate, deployment success, or neural retrieval quality is claimed by this validation record.
