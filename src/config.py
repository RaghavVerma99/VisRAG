"""Application settings — the single source of truth for configuration.

Field names map 1:1 onto the variables in ``.env.example`` (case-insensitive),
so the template *is* the schema. Nothing here reads ``os.environ`` directly.

Secrets are held as :class:`~pydantic.SecretStr` so they never leak into logs,
tracebacks or the config hash.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CaptionProvider = Literal["openai", "anthropic", "local-llava"]
QdrantMode = Literal["embedded", "server"]
Device = Literal["cpu", "cuda", "mps"]

#: Fields excluded from :meth:`Settings.config_hash` — secrets must not influence it.
_SECRET_FIELDS: set[str] = {"openai_api_key", "anthropic_api_key"}


class Settings(BaseSettings):
    """Runtime configuration, loaded from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # --- Paths -------------------------------------------------------------
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    data_eval: Path = Path("data/eval_set")

    # --- Vector store ------------------------------------------------------
    qdrant_mode: QdrantMode = "embedded"
    qdrant_path: Path = Path(".qdrant")
    qdrant_url: str | None = None
    qdrant_collection: str = Field(default="elements_v1", min_length=1)

    # --- Models ------------------------------------------------------------
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_model: str = "openai/clip-vit-base-patch32"
    device: Device = "cpu"

    caption_provider: CaptionProvider = "openai"
    caption_model: str = "gpt-4o-mini"
    caption_local_model: str = "llava-hf/llava-1.5-7b-hf"
    #: Sending document crops to a hosted VLM needs explicit approval (root §9).
    caption_allow_hosted: bool = False

    # --- Generation --------------------------------------------------------
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    vision_model: str = "gpt-4o"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_images: int = Field(default=4, ge=0)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)

    # --- Retrieval ---------------------------------------------------------
    top_k: int = Field(default=8, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rerank_candidates: int = Field(default=16, ge=1)
    abstain_threshold: float = Field(default=0.30, ge=0.0, le=1.0)

    # --- Cache -------------------------------------------------------------
    cache_enabled: bool = True
    cache_similarity: float = Field(default=0.95, ge=0.0, le=1.0)
    cache_ttl_s: int = Field(default=86_400, ge=0)
    cache_version: int = Field(default=1, ge=0)

    # --- Reproducibility ---------------------------------------------------
    seed: int = 42

    @field_validator("openai_api_key", "anthropic_api_key", "qdrant_url", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """An empty value in ``.env`` means "not configured", not an empty string."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.qdrant_mode == "server" and not self.qdrant_url:
            raise ValueError("QDRANT_MODE=server requires QDRANT_URL")

        if self.rerank_candidates < self.top_k:
            raise ValueError(
                f"RERANK_CANDIDATES ({self.rerank_candidates}) must be >= "
                f"TOP_K ({self.top_k}); reranking a set smaller than the result "
                "set silently drops candidates"
            )

        return self

    @property
    def is_vision_configured(self) -> bool:
        """True if a hosted vision model can be called."""
        return bool(self.openai_api_key or self.anthropic_api_key)

    def config_hash(self) -> str:
        """Short, stable hash of the non-secret configuration.

        Recorded in every eval report (ADR-007) so two runs can be compared
        without diffing the whole environment.
        """
        payload = self.model_dump(mode="json", exclude=_SECRET_FIELDS)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()
