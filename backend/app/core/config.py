"""Typed, environment-driven configuration.

Every swappable component (NFR-9) reads its knobs from here. Nested groups map
to env vars with a double-underscore delimiter, e.g. ``EMBEDDING__DEVICE=mps``.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production", "test"]
Device = Literal["auto", "mps", "cuda", "cpu"]

# Repo root: backend/app/core/config.py -> backend/app/core -> backend/app -> backend -> repo
REPO_ROOT = Path(__file__).resolve().parents[3]


def anchor_path(value: Path | str) -> Path:
    """Resolve a configured path against the repo root, not the process CWD.

    `.env` naturally contains relative paths like `./data/models`. The API runs
    from `backend/`, Celery from the repo root, and scripts from wherever the
    user happens to be — so a CWD-relative path silently points at three
    different directories, and the 2.3 GB model cache gets re-downloaded per
    working directory.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


class AppSettings(BaseModel):
    env: Environment = "development"
    debug: bool = True
    name: str = "PolyDoc"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    session_secret: str = "dev-only-insecure-secret"

    @property
    def is_production(self) -> bool:
        return self.env == "production"


class StorageSettings(BaseModel):
    backend: Literal["local", "s3"] = "local"
    local_dir: Path = REPO_ROOT / "data" / "uploads"
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str = "ap-south-1"

    @field_validator("local_dir", mode="before")
    @classmethod
    def _anchor(cls, v: object) -> object:
        return anchor_path(v) if isinstance(v, (str, Path)) else v


class LimitSettings(BaseModel):
    """FR-2: upload guardrails."""

    max_file_mb: int = 50
    max_pages: int = 300
    max_files_per_session: int = 25

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


class DatabaseSettings(BaseModel):
    url: str = "postgresql+asyncpg://polydoc:polydoc@localhost:5432/polydoc"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_pre_ping: bool = True

    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith("sqlite")

    @property
    def sync_url(self) -> str:
        """Driver-stripped URL for Alembic / Celery workers using sync engines."""
        return self.url.replace("+asyncpg", "+psycopg").replace("+aiosqlite", "")


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"
    celery_broker_db: int = 1
    celery_result_db: int = 2

    def _with_db(self, db: int) -> str:
        base, _, _ = self.url.rpartition("/")
        return f"{base or self.url}/{db}"

    @property
    def celery_broker_url(self) -> str:
        return self._with_db(self.celery_broker_db)

    @property
    def celery_result_url(self) -> str:
        return self._with_db(self.celery_result_db)


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection: str = "polydoc_chunks"
    prefer_grpc: bool = False
    timeout: int = 60


class EmbeddingSettings(BaseModel):
    """§8.3: bge-m3 gives dense + learned-sparse vectors from a single pass."""

    model_name: str = "BAAI/bge-m3"
    device: Device = "auto"
    use_fp16: bool = True
    max_length: int = 1024
    batch_size: int = 8
    dense_dim: int = 1024
    cache_dir: Path = REPO_ROOT / "data" / "models"

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _anchor(cls, v: object) -> object:
        return anchor_path(v) if isinstance(v, (str, Path)) else v


class RerankerSettings(BaseModel):
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: Device = "auto"
    use_fp16: bool = True
    batch_size: int = 8
    max_length: int = 1024


class RetrievalSettings(BaseModel):
    top_k: int = 20
    top_n: int = 5
    rrf_k: int = 60
    hybrid_enabled: bool = True
    min_rerank_score: float = -2.0
    dual_query_hinglish: bool = True

    # Document-level queries ("summary do", "kis baare mein hai") are served by
    # a coverage sample spanning the documents rather than by relevance ranking,
    # which has nothing to rank them against. `summary_max_chunks` is the token
    # budget for that sample; `summary_scan_limit` caps how many chunks are
    # scanned to build it, so a 300-page upload cannot pull its whole text into
    # memory on one question.
    summary_max_chunks: int = 24
    summary_scan_limit: int = 2000

    # Fusion weights. Measured on a bilingual probe set, equal-weight RRF scored
    # *worse* than dense alone (57% vs 100% recall@1): lexical overlap cannot
    # exist between a Hindi query and an English passage, so the sparse arm
    # contributes noise on exactly the cross-lingual queries this product is
    # built for. Sparse still earns its place on exact terms — clause IDs,
    # numbers, names — so it is down-weighted rather than removed.
    dense_weight: float = 0.75
    sparse_weight: float = 0.25

    # Load the embedder and reranker at startup rather than on the first query.
    # Metal/CUDA compile their kernels on first use: measured cold, retrieval
    # took 14.8s and settled at 0.37s by the fifth call. Without this, the first
    # user of every deploy pays that 40x penalty.
    warmup_on_startup: bool = True

    @field_validator("dense_weight", "sparse_weight")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("fusion weights must be non-negative")
        return v


class ChunkingSettings(BaseModel):
    """FR-8 / §11.2: sentence-aware, 300-800 tokens, 10-15% overlap."""

    target_tokens: int = 550
    min_tokens: int = 200
    max_tokens: int = 800
    overlap_ratio: float = 0.12

    @field_validator("overlap_ratio")
    @classmethod
    def _sane_overlap(cls, v: float) -> float:
        if not 0.0 <= v < 0.5:
            raise ValueError("overlap_ratio must be in [0.0, 0.5)")
        return v


class OCRSettings(BaseModel):
    """FR-3/FR-4 and the §11.1 text-density routing rule."""

    engine: Literal["tesseract", "none"] = "tesseract"
    langs: str = "hin+eng"
    dpi: int = 300
    min_chars_per_page: int = 50
    preprocess: bool = True
    max_workers: int = 4


LLMProvider = Literal["gemini", "anthropic", "mock"]

# Fallback model per provider, used when LLM__MODEL is left unset.
#
# Why a Flash model and not Pro: the Gemini free tier grants `limit: 0` for the
# Pro models, and Flash is the right shape for grounded RAG anyway — retrieval
# does the heavy lifting, and the lower latency is what keeps us inside the
# NFR-2 P95 budget. On a billed key, `LLM__MODEL=gemini-2.5-pro` is a one-line
# upgrade if the eval numbers call for it.
DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-3-flash-preview",
    "anthropic": "claude-opus-5",
    "mock": "mock-echo",
}

# Tried in order by `scripts/check_llm.py` when the configured model is
# quota-blocked or retired, so a key still lands on something usable.
# Note: gemini-2.5-flash now 404s for projects created after its cutoff
# ("no longer available to new users"), hence the -3 preview leading here.
GEMINI_FALLBACK_ORDER: tuple[str, ...] = (
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
)

# Env var each provider's credential is read from. Kept as the conventional
# name so the vendor SDKs pick it up without us re-exporting anything.
PROVIDER_KEY_ENV: dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mock": "",
}


class LLMSettings(BaseModel):
    """Generation config (§8.3). The provider is swappable by design — the PRD
    names Gemini as a sanctioned alternate to Claude."""

    provider: LLMProvider = "gemini"
    model: str = ""
    max_tokens: int = 4096

    temperature: float = 0.2
    # Reasoning depth. Mapped per provider in the generation layer: Gemini
    # thinking_budget, Anthropic output_config.effort.
    effort: Literal["low", "medium", "high"] = "medium"
    thinking: bool = False
    timeout_seconds: float = 120.0
    max_retries: int = 2

    # Per-million-token prices for the NFR-4 cost cap. Defaults are Gemini
    # Flash-tier list prices; override per provider/model in .env.
    input_cost_per_mtok: float = 0.30
    output_cost_per_mtok: float = 2.50
    #: Refuse to answer above this per-query cost, 0 disables (NFR-4).
    max_cost_per_query_usd: float = 0.0

    @field_validator("model", mode="before")
    @classmethod
    def _strip_inline_comment(cls, v: object) -> object:
        """`.env` is hand-edited, and dotenv keeps the trailing `# ...` when a
        key is left blank. Treat a comment-only value as unset rather than
        letting it reach the provider as a model name."""
        if isinstance(v, str):
            v = v.split("#", 1)[0].strip()
        return v

    @property
    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, "")

    @property
    def api_key_env(self) -> str:
        return PROVIDER_KEY_ENV.get(self.provider, "")


class ObservabilitySettings(BaseModel):
    log_level: str = "INFO"
    json_logs: bool = False
    metrics_enabled: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    obs: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    # Read from the conventional env var names so the vendor SDKs and our
    # settings agree on one source of truth.
    google_api_key: str | None = None
    anthropic_api_key: str | None = None

    @property
    def llm_api_key(self) -> str | None:
        """Credential for the currently selected provider."""
        return {
            "gemini": self.google_api_key,
            "anthropic": self.anthropic_api_key,
        }.get(self.llm.provider)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def _pin_model_cache() -> None:
    """Point the HuggingFace cache at the configured directory.

    FlagEmbedding does not reliably forward `cache_dir` to every download, so
    weights would otherwise land in `~/.cache/huggingface` — outside the Docker
    volume, meaning every container rebuild re-downloads 2.5 GB and the image
    cannot be shipped warm.
    """
    import os

    cache_dir = settings.embedding.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir))


_pin_model_cache()
