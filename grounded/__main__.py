import argparse
import json
import logging
from pathlib import Path

from .config import Settings
from .models import Query
from .service import RAGService


def main():
    parser = argparse.ArgumentParser(prog="python -m grounded")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Start the web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    ingest = sub.add_parser("ingest", help="Index a file or a directory")
    ingest.add_argument("path", type=Path)
    ask = sub.add_parser("ask", help="Ask a standalone question")
    ask.add_argument("question")
    ask.add_argument("--strategy", choices=["hybrid", "bm25", "dense"], default="hybrid")
    sub.add_parser("documents", help="List indexed documents")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "serve":
        import uvicorn
        uvicorn.run("grounded.api:create_app", factory=True, host=args.host, port=args.port)
        return
    service = RAGService(Settings())
    try:
        if args.command == "ingest":
            if not args.path.exists():
                parser.error(f"Path does not exist: {args.path}")
            paths = sorted(p for p in args.path.rglob("*") if p.is_file()
                           and p.suffix.lower() in {".txt", ".md", ".pdf"}) if args.path.is_dir() else [args.path]
            if not paths:
                parser.error("No PDF, Markdown, or text files found")
            for path in paths:
                if path.stat().st_size > service.settings.max_upload_bytes:
                    parser.error(f"File exceeds the upload size limit: {path.name}")
                print(json.dumps({"filename": path.name, **service.ingest(path.name, path.read_bytes())}))
        elif args.command == "ask":
            result = service.ask(Query(question=args.question, strategy=args.strategy,
                                       top_k=service.settings.top_k))
            print(result.model_dump_json(indent=2))
        elif args.command == "documents":
            print(json.dumps(service.store.documents(), indent=2))
    finally:
        service.provider.close()


if __name__ == "__main__":
    main()
