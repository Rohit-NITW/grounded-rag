"""BM25 + cosine search, fused using reciprocal ranks (RRF)."""

import re
from collections import Counter
from dataclasses import dataclass
from math import log

import numpy as np

STOPWORDS = set("a an the is are was were be been being to of for on in at as by with and or "
                "but if it its this that these those do does did how what which who when where "
                "why can could should would will i we you they our your their me my us please "
                "tell about from have has had may must much many".split())


def tokenize(text: str) -> list[str]:
    return [word for word in re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", text.casefold())
            if word not in STOPWORDS]


def bm25_scores(question: str, documents: list[str]) -> np.ndarray:
    counts = [Counter(tokenize(text)) for text in documents]
    lengths = np.array([sum(c.values()) for c in counts], dtype=float)
    average = max(float(lengths.mean()), 1.0) if len(lengths) else 1.0
    scores = np.zeros(len(counts))
    for term in set(tokenize(question)):
        df = sum(term in c for c in counts)
        idf = log(1 + (len(counts) - df + 0.5) / (df + 0.5))
        tf = np.array([c[term] for c in counts], dtype=float)
        scores += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths / average))
    return scores


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] = scores.get(index, 0) + 1 / (k + rank)
    return scores


@dataclass
class Hit:
    row: dict
    score: float
    cosine: float
    bm25: float


def retrieve(question: str, rows: list[dict], vectors: np.ndarray, query_vector: np.ndarray | None,
             top_k: int, strategy: str, neural: bool, min_similarity: float) -> list[Hit]:
    if not rows or not tokenize(question):
        return []
    lexical = bm25_scores(question, [r["text"] for r in rows])
    cosine = np.zeros(len(rows))
    if query_vector is not None:
        if vectors.shape[1] != len(query_vector):
            raise ValueError("Embedding dimensions differ; re-ingest with the current model")
        # Providers normalize; normalize again to make the retrieval boundary explicit.
        cosine = (vectors @ query_vector) / np.maximum(
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector), 1e-12)
    lexical_order = [int(i) for i in np.argsort(-lexical, kind="stable") if lexical[i] > 0][:30]
    dense_order = [int(i) for i in np.argsort(-cosine, kind="stable") if cosine[i] > 0][:30]
    if strategy == "bm25":
        scores = {i: float(lexical[i]) for i in lexical_order}
    elif strategy == "dense":
        scores = {i: float(cosine[i]) for i in dense_order}
    elif strategy == "hybrid":
        scores = reciprocal_rank_fusion([lexical_order, dense_order])
    else:
        raise ValueError("Unknown retrieval strategy")
    terms = set(tokenize(question))
    results = []
    for i in sorted(scores, key=lambda i: (-scores[i], rows[i]["id"])):
        coverage = len(terms & set(tokenize(rows[i]["text"]))) / len(terms)
        # A transparent heuristic, not a calibrated confidence score.
        lexical_evidence = lexical[i] > 0 and coverage >= 0.25
        semantic_evidence = neural and cosine[i] >= min_similarity and strategy != "bm25"
        if not (lexical_evidence or semantic_evidence):
            continue
        results.append(Hit(rows[i], scores[i], float(cosine[i]), float(lexical[i])))
        if len(results) == top_k:
            break
    return results
