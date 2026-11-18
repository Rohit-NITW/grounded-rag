from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Query(StrictModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    strategy: Literal["hybrid", "bm25", "dense"] = "hybrid"
    document_ids: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("question")
    @classmethod
    def trim_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Enter a question with at least two characters")
        return value


class Claim(StrictModel):
    text: str = Field(min_length=1, max_length=4000)
    source_ids: list[str] = Field(min_length=1, max_length=10)


class GeneratedAnswer(StrictModel):
    insufficient_evidence: bool
    claims: list[Claim] = Field(max_length=10)


class Source(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    filename: str
    page: int
    text: str
    score: float
    cosine: float
    bm25: float


class Answer(BaseModel):
    question: str
    answer: str
    claims: list[Claim]
    sources: list[Source]
    insufficient_evidence: bool
    reason: str | None = None
    provider: str
    model: str
    strategy: str
    timings_ms: dict[str, float]
    usage: dict[str, int]
    request_id: str = ""
