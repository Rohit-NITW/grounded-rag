"""Run against a real configured provider. OpenAI mode incurs small API charges."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounded.config import Settings
from grounded.models import Query
from grounded.service import RAGService


def main():
    with tempfile.TemporaryDirectory(prefix="grounded-live-") as folder:
        settings = Settings(data_dir=Path(folder))
        if settings.provider == "demo":
            raise SystemExit("Configure RAG_PROVIDER=ollama or openai in .env before this live test.")
        service = RAGService(settings)
        try:
            service.ingest("smoke.md", b"In the fictional smoke-test policy, backups are retained for exactly 17 days.")
            answer = service.ask(Query(question="For exactly how many days are backups retained?"))
            if answer.insufficient_evidence or not answer.claims or "17" not in answer.answer:
                raise AssertionError("The model did not produce the expected supported answer. Inspect model configuration and output.")
            print(json.dumps({"status": "passed", "provider": settings.provider,
                              "model": settings.chat_model, "answer": answer.answer,
                              "timings_ms": answer.timings_ms, "usage": answer.usage}, indent=2))
        finally:
            service.provider.close()


if __name__ == "__main__":
    main()
