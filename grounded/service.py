import hashlib
import time

from .config import Settings
from .ingestion import DocumentError, make_chunks, parse_document
from .models import Answer, Query, Source
from .providers import Provider
from .retrieval import retrieve
from .store import Store

ABSTENTION = "I couldn't find enough evidence in the selected documents to answer that."


class RAGService:
    def __init__(self, settings: Settings, provider: Provider | None = None):
        self.settings = settings
        self.store = Store(settings.db_path, settings.max_chunks)
        self.provider = provider or Provider(settings)

    def ingest(self, filename: str, content: bytes) -> dict:
        if len(content) > self.settings.max_upload_bytes:
            raise DocumentError("File exceeds the upload size limit")
        digest = hashlib.sha256(content).hexdigest()
        if self.store.contains(digest):
            return {"document_id": digest, "duplicate": True, "chunks_added": 0}
        pages = parse_document(filename, content, self.settings.max_document_chars)
        document_id, chunks = make_chunks(filename, content, pages, self.settings.chunk_words,
                                          self.settings.chunk_overlap)
        if len(chunks) > self.settings.max_chunks:
            raise DocumentError("This document exceeds the index capacity")
        vectors = self.provider.embed([c.text for c in chunks])
        added = self.store.add(document_id, chunks, vectors, len(content))
        return {"document_id": document_id, "duplicate": not added,
                "chunks_added": len(chunks) if added else 0}

    def search(self, query: Query) -> list[Source]:
        rows, vectors = self.store.snapshot(query.document_ids)
        if not rows:
            return []
        query_vector = None
        if query.strategy != "bm25":
            query_vector = self.provider.embed([query.question], query=True)[0]
        hits = retrieve(query.question, rows, vectors, query_vector, query.top_k,
                        query.strategy, self.settings.provider != "demo", self.settings.min_similarity)
        return [Source(source_id=f"S{i}", chunk_id=hit.row["id"],
                       document_id=hit.row["document_id"], filename=hit.row["filename"],
                       page=hit.row["page"], text=hit.row["text"], score=round(hit.score, 6),
                       cosine=round(hit.cosine, 6), bm25=round(hit.bm25, 6))
                for i, hit in enumerate(hits, 1)]

    def ask(self, query: Query) -> Answer:
        start = time.perf_counter()
        sources = self.search(query)
        retrieved = time.perf_counter()
        claims = []
        usage = {}
        reason = None
        if not sources:
            reason = "no_relevant_sources"
        else:
            generated, usage = self.provider.generate(query.question, sources)
            allowed_ids = {source.source_id for source in sources}
            if generated.insufficient_evidence or not generated.claims:
                reason = "model_abstained"
            elif any(not set(claim.source_ids) <= allowed_ids for claim in generated.claims):
                reason = "invalid_citations"
            else:
                claims = generated.claims
        finished = time.perf_counter()
        answer = "\n\n".join(c.text + " " + " ".join(f"[{s}]" for s in c.source_ids) for c in claims)
        return Answer(
            question=query.question, answer=answer or ABSTENTION, claims=claims, sources=sources,
            insufficient_evidence=bool(reason), reason=reason, provider=self.settings.provider,
            model="extractive-preview" if self.settings.provider == "demo" else self.settings.chat_model,
            strategy=query.strategy,
            timings_ms={"retrieval": round((retrieved - start) * 1000, 2),
                        "generation": round((finished - retrieved) * 1000, 2),
                        "total": round((finished - start) * 1000, 2)}, usage=usage,
        )
