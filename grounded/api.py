import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .ingestion import DocumentError
from .models import Answer, Query
from .providers import ProviderError
from .service import RAGService

logger = logging.getLogger("grounded")


class BodyLimitMiddleware:
    """Cap actual bytes before multipart parsing, including chunked request bodies."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        parts, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            total += len(body)
            if total > self.limit:
                response = JSONResponse({"detail": "Request exceeds the upload size limit"}, status_code=413)
                return await response(scope, receive, send)
            parts.append(body)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(parts), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    settings = settings or Settings()
    service = service or RAGService(settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.provider.close()

    app = FastAPI(title="Grounded RAG API", version="1.0.0", lifespan=lifespan,
                  description="Single-user document research assistant. See README for provider setup.")
    app.state.service = service
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_upload_bytes + 65_536)

    @app.middleware("http")
    async def request_metadata(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            )
        logger.info(json.dumps({"request_id": request.state.request_id,
                               "method": request.method, "status": response.status_code,
                               "duration_ms": round((time.perf_counter() - start) * 1000, 2)}))
        return response

    def authorize(authorization: str | None = Header(default=None)):
        token = settings.api_token.get_secret_value()
        if token and not hmac.compare_digest(authorization or "", "Bearer " + token):
            raise HTTPException(401, "A valid access token is required", headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(DocumentError)
    async def document_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(ProviderError)
    async def provider_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/api/config", dependencies=[Depends(authorize)])
    def config():
        return {"provider": settings.provider,
                "model": "extractive-preview" if settings.provider == "demo" else settings.chat_model,
                "embedding_model": "hash-v1-768" if settings.provider == "demo" else settings.embed_model,
                "index_id": settings.index_id, "top_k": settings.top_k,
                "max_upload_mb": settings.max_upload_bytes // (1024 * 1024)}

    @app.get("/api/documents", dependencies=[Depends(authorize)])
    def documents():
        return {"documents": service.store.documents()}

    @app.post("/api/documents", dependencies=[Depends(authorize)], status_code=201)
    async def upload(file: UploadFile):
        try:
            content = await file.read(settings.max_upload_bytes + 1)
            if len(content) > settings.max_upload_bytes:
                raise HTTPException(413, "File exceeds the upload size limit")
            return await run_in_threadpool(service.ingest, file.filename or "document.txt", content)
        finally:
            await file.close()

    @app.delete("/api/documents/{document_id}", dependencies=[Depends(authorize)])
    def delete_document(document_id: str):
        if not service.store.delete(document_id):
            raise HTTPException(404, "Document not found")
        return {"deleted": True}

    @app.post("/api/search", dependencies=[Depends(authorize)])
    def search(query: Query):
        return {"sources": service.search(query)}

    @app.post("/api/ask", response_model=Answer, dependencies=[Depends(authorize)])
    def ask(query: Query, request: Request):
        result = service.ask(query)
        result.request_id = request.state.request_id
        return result

    web = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(web / "index.html")

    return app
