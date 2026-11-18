# Presenting Grounded as a portfolio project

## Project entry

**Grounded — Document Research Assistant**  
Python, FastAPI, SQLite, NumPy, Ollama/OpenAI, Docker

Use these bullets after running the project, understanding the implementation, and making it your own:

- Built a document question-answering application with PDF/Markdown ingestion, hybrid BM25 and vector retrieval, and source-linked answers through local or hosted LLM adapters.
- Implemented persistent indexing, duplicate detection, document filtering, and citation-ID validation, with 42 automated tests covering retrieval, APIs, storage, and provider failure cases.
- Created a 24-question evaluation harness comparing BM25, vector, and hybrid retrieval using evidence Recall@k, MRR, abstention specificity, and latency.

Only claim Docker deployment, live LLM evaluation, or public hosting after you perform them. The supplied Docker and CI files are configurations; they are not proof of a successful deployment or workflow run.

## Quantified bullet after your own experiments

Replace every bracket with a number you actually measure:

> Evaluated [model + embedding model] over [N] documents and [M] held-out questions; achieved [R] evidence Recall@5, [A] manually reviewed answer correctness, and [T] ms p95 response latency on [hardware], compared with [baseline].

Do not turn the bundled demo's 100% retrieval recall into “100% LLM accuracy.” The packaged run uses lexical hashes, only six small fictional documents, and questions written from the same corpus. Its unanswerable refusal rate is 37.5%, and hybrid ties BM25 on MRR. Those limitations are useful interview discussion material.

## Five-minute demonstration

1. **0:00–0:40 — Problem.** “Teams have answers spread across runbooks and guides. I built an assistant that retrieves the relevant passages and lets users inspect the evidence.” Show the provider badge so the audience knows whether this is an LLM or extractive preview.
2. **0:40–1:30 — Happy path.** Ask “What is the procedure for a production rollback?” Click a citation and show the matching passage and page number.
3. **1:30–2:10 — A user-supplied file.** Upload a short document you wrote. Ask a question answerable only from that file. Select it in the source filter.
4. **2:10–2:50 — Failure case.** Ask for a detail absent from the corpus. Explain any abstention or wrong response. Distinguish retrieval relevance from answer support.
5. **2:50–3:40 — Architecture.** Show the ingestion, retrieval, generation, and citation-validation modules. Explain why the storage is SQLite and why RRF combines ranks.
6. **3:40–4:30 — Evidence.** Run the tests and show the evaluation JSON. State corpus size, provider, test split, and actual limitations.
7. **4:30–5:00 — Your contribution.** Demonstrate one extension you designed, explain the tradeoff, and show how you measured it.

## Questions to be ready for

| Interview question | Points to explain |
|---|---|
| What makes this RAG? | Retrieval selects external context before generation; the model is not trained on the documents. |
| Why use both BM25 and vectors? | Keywords help with exact identifiers; neural embeddings can help paraphrases; fusion must still be evaluated. |
| Why RRF instead of adding raw scores? | BM25 and cosine scales differ; RRF combines ranking positions without a score calibration step. |
| Why overlap chunks? | It reduces the risk of losing evidence at boundaries, but adds storage and duplicate context. |
| How do you prevent hallucinations? | Evidence-only prompts, abstention, and reference validation help; none guarantees factual correctness. |
| Are citations verified? | IDs are checked against the supplied context. Claim entailment is not verified automatically. |
| Why no vector database? | Exact scans make a small demo reproducible and easy to inspect. Large indexes need an indexed search design. |
| How do you handle updates? | Different bytes create a new document ID; obsolete versions must be removed explicitly. |
| Why separate indexes per embedding model? | Vectors from different embedding spaces cannot be compared meaningfully. |
| What does 100% Recall@5 mean? | All required labeled passages were found in that specific small test; it says nothing directly about answer correctness. |
| What happens if the provider fails? | The API returns a safe error. Failed ingestion does not leave partial database records. |
| What would you improve first? | Choose a measured bottleneck: retrieval quality, rejection of unanswerable questions, ingestion throughput, or authorization. |

## Make the work yours

**First:** run demo mode, inspect `service.py`, and trace one upload and query through each function. Then run actual Ollama or OpenAI mode and verify an answer with citations.

**Next:** replace the sample corpus with material you have permission to use: your own project docs, open-source architecture docs, or course notes. Write at least 50 realistic test questions, including questions that cannot be answered. Reserve a held-out subset before tuning.

**Then:** add one substantive feature, such as reranking or asynchronous ingestion. Record before/after metrics and at least two cases where the change helps or hurts. Explain those outcomes in the README.

**Finally:** publish the repository under your account, add your screenshots and short demo recording, and include only your measured results. Be candid about AI assistance if asked, and be able to explain or modify the code yourself.

## Suggested repository description

“Inspectable RAG assistant with local/hosted LLMs, hybrid retrieval, source citations, persistent document indexing, and a reproducible evaluation harness.”

Suggested topics: `rag`, `llm`, `fastapi`, `ollama`, `retrieval`, `bm25`, `embeddings`, `evaluation`, `python`.
