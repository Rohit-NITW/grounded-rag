# Design and implementation notes

## Problem

A normal chat model does not know the contents of a user's local project documents. Pasting whole files into every question is wasteful and makes sources hard to inspect. Grounded retrieves a small set of potentially relevant passages, supplies them to a model, and returns an answer alongside that exact context.

RAG changes the context supplied at inference time. This project does not train or fine-tune a model.

## Ingestion

1. Enforce upload byte limits; validate the supported extension and decode UTF-8 or parse PDF pages.
2. Preserve PDF page numbers. Text and Markdown documents use page 1.
3. Normalize whitespace, then create windows of at most 220 words with 40-word overlap by default. A 6,000-character secondary cap bounds unusually long chunks. Reject single unbroken strings above that cap.
4. Hash the original bytes with SHA-256. Identical bytes uploaded again reuse the first indexed document, even under a different filename.
5. Embed batches of at most 32 chunks. Normalize every vector to unit L2 length.
6. Commit document metadata, chunks, and vectors in one SQLite transaction. A failed embedding request leaves no partially indexed document.

The original binary upload is not retained. The index stores extracted text, vectors, sanitized filename, page, word offset, and timestamps. The evidence viewer therefore displays extracted passages, not a visual PDF highlight.

Changed bytes create a new document version with a new ID. This is not an in-place update: explicitly delete obsolete versions to avoid retrieving conflicting copies.

## Persistence and identity

SQLite uses foreign keys and WAL mode. Each operation owns its connection. `BEGIN IMMEDIATE` serializes the duplicate/capacity check and insertion. `ON DELETE CASCADE` removes chunks with the document. Each query retrieves a consistent snapshot using one SELECT, then scores it outside the database connection.

Vectors are JSON arrays in SQLite for readability and easy portability. This is less compact and slower to decode than a packed binary representation or a dedicated vector index. That tradeoff is acceptable for the small corpus this version targets.

The index filename includes a fingerprint of the provider, embedding model, and chunk configuration. This avoids mixing incompatible embeddings. The database also rejects a changed embedding dimension during insertion. It cannot detect a provider changing model weights while retaining the same name and dimension; pin models and use a new data directory when that happens.

## Retrieval

**BM25:** tokenize Unicode words, remove a small English stopword set, and compute BM25 with `k1=1.5`, `b=0.75`. Document frequency and average passage length are computed over the selected corpus. No stemming or multilingual normalization is implemented.

For term `t` and passage `d`:

```text
IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
BM25(t,d) = IDF(t) × tf(t,d) × (k1 + 1)
             / (tf(t,d) + k1 × (1 - b + b × len(d) / avg_len))
```

**Vector search:** embed the question, then compute exact cosine similarity against every selected passage. In real-provider modes these are neural embeddings. Demo mode uses stable signed feature hashing of tokens into 768 dimensions; it has lexical overlap behavior and hash collisions, not semantic language understanding.

**Hybrid search:** take up to 30 positive-score candidates from each branch. Add `1 / (60 + rank)` for each appearance in either ranking. Sort by the fused score, with stable chunk IDs breaking ties. RRF combines rank positions because BM25 values and cosine values have different scales.

**Relevance gate:** a passage is eligible when it contains at least 25% of the distinct query terms and has a positive BM25 score, or when a real neural provider gives cosine similarity at least `RAG_MIN_SIMILARITY`. BM25-only mode uses the lexical condition. Demo-vector mode also uses the lexical gate to prevent accepting pure hash collisions.

The lexical gate in vector-only demo mode means this is a comparison of scoring strategies with a shared eligibility rule, not a completely independent dense-only system. The benchmark reports the same implementation users run.

Scores are ranking signals, not probabilities. The defaults are heuristics. They are not calibrated across embedding models or domains. Low lexical coverage can reject valid paraphrases; common words can admit irrelevant passages. The generator's abstention decision is a second, fallible check.

**Cost:** with `n` passages and embedding dimension `d`, cosine scoring is O(n×d); BM25 also rebuilds term statistics per query. There is no ANN index or query cache. Exact scans and no cache simplify persistence/deletion behavior, but do not scale to very large corpora.

## Generation and citations

The provider receives a system instruction plus a JSON-serialized question and evidence array. Evidence is explicitly described as untrusted data. Each source gets a short ID such as `S1`. The model must return:

```json
{
  "insufficient_evidence": false,
  "claims": [
    {"text": "Backups are retained for thirty days.", "source_ids": ["S1"]}
  ]
}
```

Pydantic validates the returned structure. The service rejects the whole generated answer if any claim uses an ID absent from the provided sources. Empty claims or a model abstention return a standard insufficient-evidence message. Invalid JSON or an upstream failure becomes an actionable HTTP 502; it never silently falls back to demo output.

This checks reference integrity, not entailment. A source ID may exist while the associated claim is unsupported. The system has no automatic factual correctness guarantee. Prompt injection can still affect the model; it has no tools or actions to invoke in this application.

Ollama calls `/api/embed` and `/api/chat`. OpenAI calls `/v1/embeddings` and `/v1/responses`. Transient HTTP statuses get bounded retries for embedding calls only; network/validation failures and generation calls are not automatically retried. Timeouts apply per HTTP operation, not to the entire ingestion job.

## API and interface

The UI and REST API are served by the same FastAPI app. No CORS grant is needed. Model keys never reach the browser. An optional static bearer token protects all `/api/*` routes. The UI keeps it in memory, not local storage. It is a single shared workspace, not an authorization boundary between users.

Uploads run parsing and embedding in the thread pool. Ordinary blocking service calls use FastAPI's synchronous endpoint thread handling. SQLite transactions protect storage correctness but there is no background ingestion queue, cancellation, rate limiter, or global concurrency budget.

Request logs contain generated request ID, method, status, and elapsed time, not the question or document body. Responses expose retrieval/generation timing and available generation token usage. Embedding token usage is not aggregated. There is no external telemetry exporter.

The frontend uses `textContent` and DOM creation for user content. It does not render model Markdown/HTML, preventing source text from becoming executable HTML in this UI. A restrictive CSP applies to the app page and static assets. Swagger UI has its framework-provided behavior and assets.

## Evaluation design

The 24-case fixture is a small development set written with the fictional corpus. It contains positive questions, multi-evidence questions, off-topic negatives, and near-topic negatives. Gold evidence is defined as filename + required phrase. Each evaluation run gets a new temporary index, preventing unrelated uploaded documents from contaminating results.

Evidence Recall@k measures retrieval coverage, MRR measures the position of the first useful passage, and abstention specificity measures rejection of questions without an answer. Citation validity is structurally guaranteed for displayed answers and is not a substitute for a human grounding score. p50 and p95 timings are observations from one run, not controlled load-test results.

The included results show no hybrid improvement over BM25 on this tiny lexical fixture. A useful experiment is to add paraphrases, acronyms, conflicting versions, and difficult near-topic negatives to a held-out set, then compare neural and lexical retrieval.

## What a larger deployment would change

| Concern | Current decision | Larger deployment |
|---|---|---|
| Vector search | Exact NumPy scan | ANN index such as pgvector/HNSW, with measured recall tradeoffs |
| Ingestion | Request-scoped work | Durable queue, worker pool, progress and retry state |
| Authorization | Optional shared token | Per-user authentication and per-document grants before retrieval |
| Parsing | In-process bounded parsing | Isolated workers with CPU, memory, and wall-clock limits |
| Quality | RRF + heuristic gate | Reranking, calibrated thresholds, held-out claim-support review |
| Observability | JSON logs and timings | Central metrics, traces, error budgets, cost accounting |
| Deletion | Logical indexed-row deletion | Lifecycle policy covering uploads, indexes, journals, and backups |

These are extension ideas, not implemented features.
