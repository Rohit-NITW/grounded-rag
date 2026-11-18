"""Reproducible retrieval ablation; optional generation checks are not factual grading."""

import argparse
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import Settings
from .models import Query
from .service import RAGService


def evaluate(service: RAGService, cases: list[dict], top_k: int, generate: bool) -> dict:
    report = {}
    for strategy in ("bm25", "dense", "hybrid"):
        details = []
        for case in cases:
            query = Query(question=case["question"], top_k=top_k, strategy=strategy)
            start = time.perf_counter()
            answer = service.ask(query) if generate else None
            sources = answer.sources if answer else service.search(query)
            elapsed = (time.perf_counter() - start) * 1000
            gold = case["evidence"]
            matched = [any(s.filename == e["filename"] and e["contains"].casefold() in s.text.casefold()
                           for s in sources) for e in gold]
            ranks = [i for i, s in enumerate(sources, 1) if any(
                s.filename == e["filename"] and e["contains"].casefold() in s.text.casefold() for e in gold)]
            row = {"id": case["id"], "question": case["question"], "answerable": bool(gold),
                   "evidence_recall": sum(matched) / len(gold) if gold else None,
                   "reciprocal_rank": 1 / min(ranks) if ranks else 0,
                   "retrieval_abstained": not sources,
                   "negative_type": case.get("negative_type"),
                   "latency_ms": round(elapsed, 3), "filenames": [s.filename for s in sources]}
            if answer:
                ids = {s.source_id for s in sources}
                row.update({"answer": answer.answer, "answer_abstained": answer.insufficient_evidence,
                            "citation_valid": bool(answer.claims) and all(
                                set(c.source_ids) <= ids for c in answer.claims)})
            details.append(row)
        positives = [r for r in details if r["answerable"]]
        negatives = [r for r in details if not r["answerable"]]
        mean = lambda values: float(np.mean(values)) if values else None
        metrics = {
            f"evidence_recall_at_{top_k}": mean([r["evidence_recall"] for r in positives]),
            f"mrr_at_{top_k}": mean([r["reciprocal_rank"] for r in positives]),
            "retrieval_abstention_specificity": mean([r["retrieval_abstained"] for r in negatives]),
            "retrieval_answerable_coverage": mean([not r["retrieval_abstained"] for r in positives]),
            "latency_p50_ms": float(np.percentile([r["latency_ms"] for r in details], 50)),
            "latency_p95_ms": float(np.percentile([r["latency_ms"] for r in details], 95)),
        }
        if generate:
            answered = [r for r in details if not r["answer_abstained"]]
            metrics.update({"answer_abstention_specificity": mean([r["answer_abstained"] for r in negatives]),
                            "answerable_coverage": mean([not r["answer_abstained"] for r in positives]),
                            "citation_validity_among_answers": mean([r["citation_valid"] for r in answered])})
        report[strategy] = {"metrics": metrics, "cases": details}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/questions.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("samples"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--generate", action="store_true", help="Also call the generator; may incur API charges")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if not cases or not args.corpus.is_dir():
        parser.error("Provide a non-empty dataset and an existing corpus directory")
    # A fresh, isolated index prevents leakage from documents uploaded to the web app.
    with tempfile.TemporaryDirectory(prefix="grounded-eval-") as folder:
        settings = Settings(data_dir=Path(folder))
        service = RAGService(settings)
        try:
            for path in sorted(args.corpus.iterdir()):
                if path.suffix.lower() in {".md", ".txt", ".pdf"}:
                    service.ingest(path.name, path.read_bytes())
            report = {"created_at": datetime.now(timezone.utc).isoformat(),
                      "provider": settings.provider,
                      "embedding_model": "hash-v1-768" if settings.provider == "demo" else settings.embed_model,
                      "chat_model": "extractive-preview" if settings.provider == "demo" else settings.chat_model,
                      "index_id": settings.index_id, "python": platform.python_version(),
                      "top_k": args.top_k, "generation_enabled": args.generate,
                      "dataset_size": len(cases), "documents": len(service.store.documents()),
                      "chunk_words": settings.chunk_words, "chunk_overlap": settings.chunk_overlap,
                      "min_similarity": settings.min_similarity,
                      "note": "Synthetic development fixture, not a held-out benchmark. Demo vectors are lexical hashes. Citation validity does not measure factual accuracy.",
                      "strategies": evaluate(service, cases, args.top_k, args.generate)}
        finally:
            service.provider.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["metrics"] for k, v in report["strategies"].items()}, indent=2))
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
