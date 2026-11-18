from functools import cached_property
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", extra="ignore", populate_by_name=True
    )
    provider: Literal["demo", "ollama", "openai"] = "demo"
    data_dir: Path = Path("data")
    ollama_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen2.5:3b"
    embed_model: str = "nomic-embed-text"
    openai_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="OPENAI_API_KEY")
    api_token: SecretStr = SecretStr("")
    chunk_words: int = Field(default=220, ge=40, le=500)
    chunk_overlap: int = Field(default=40, ge=0)
    top_k: int = Field(default=5, ge=1, le=10)
    min_similarity: float = Field(default=0.35, ge=0, le=1)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    max_document_chars: int = Field(default=500_000, ge=1000)
    max_chunks: int = Field(default=10_000, ge=1)
    request_timeout: float = Field(default=120, gt=0, le=600)

    @model_validator(mode="after")
    def validate_config(self):
        if self.chunk_overlap >= self.chunk_words:
            raise ValueError("chunk_overlap must be less than chunk_words")
        if self.provider == "openai" and not self.openai_api_key.get_secret_value():
            raise ValueError("OPENAI_API_KEY is required for the openai provider")
        return self

    @cached_property
    def index_id(self) -> str:
        # Never compare vectors produced by different models or chunk configurations.
        embed = "hash-v1-768" if self.provider == "demo" else self.embed_model
        identity = f"schema-v1:{self.provider}:{embed}:{self.chunk_words}:{self.chunk_overlap}"
        return sha256(identity.encode()).hexdigest()[:16]

    @property
    def db_path(self) -> Path:
        return self.data_dir / f"index-{self.index_id}.sqlite3"
