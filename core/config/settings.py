"""
VectorLift — Application Settings
==================================
All configuration is loaded from environment variables and / or a .env file.
Nested sub-settings (Elasticsearch, Qdrant, …) use Pydantic BaseModel so that
environment variables can be namespaced (e.g. ELASTICSEARCH_HOST).

Usage
-----
    from core.config.settings import get_settings

    settings = get_settings()
    print(settings.api.port)
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AppEnv(str, Enum):
    """Runtime environment."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatasetMode(str, Enum):
    """Controls how much data is loaded / indexed."""

    DEV = "dev"       # tiny slice — fast CI / local dev
    SMALL = "small"   # ~10k passages — quick experiments
    FULL = "full"     # complete corpus — production


class RetrievalMode(str, Enum):
    """Which retrieval pipeline to activate."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    RERANK = "rerank"


class HybridFusionStrategy(str, Enum):
    """How BM25 and dense scores are combined in hybrid mode."""

    SCORE_FUSION = "score_fusion"
    RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"
    LEARNED = "learned"


# ---------------------------------------------------------------------------
# Nested settings blocks
# ---------------------------------------------------------------------------


class ElasticsearchSettings(BaseSettings):
    """Elasticsearch connection and index configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ELASTICSEARCH_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(default="localhost", description="ES host")
    port: int = Field(default=9200, ge=1, le=65535)
    index: str = Field(default="vectorlift_passages", description="Default index name")
    user: str = Field(default="elastic")
    password: SecretStr = Field(default=SecretStr(""))
    scheme: str = Field(default="http", pattern="^https?$")
    verify_certs: bool = Field(default=False)
    ca_cert: str | None = Field(default=None, description="Path to CA certificate for TLS")
    request_timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_on_timeout: bool = Field(default=True)

    @property
    def url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(default="localhost")
    port: int = Field(default=6333, ge=1, le=65535)
    collection: str = Field(default="vectorlift_dense")
    api_key: SecretStr | None = Field(default=None)
    grpc_port: int = Field(default=6334, ge=1, le=65535)
    prefer_grpc: bool = Field(default=False)
    timeout: float = Field(default=30.0, gt=0)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class PostgresSettings(BaseSettings):
    """PostgreSQL connection configuration."""

    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    db: str = Field(default="vectorlift")
    user: str = Field(default="vectorlift")
    password: SecretStr = Field(default=SecretStr("vectorlift_secret"))
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0, le=200)
    echo: bool = Field(default=False, description="Log all SQL statements")

    @property
    def async_url(self) -> str:
        pwd = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{pwd}@{self.host}:{self.port}/{self.db}"

    @property
    def sync_url(self) -> str:
        pwd = self.password.get_secret_value()
        return f"postgresql+psycopg2://{self.user}:{pwd}@{self.host}:{self.port}/{self.db}"


class RedisSettings(BaseSettings):
    """Redis cache configuration."""

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: SecretStr | None = Field(default=None)
    ttl_seconds: int = Field(default=3600, ge=0, description="Default key TTL in seconds")
    max_connections: int = Field(default=20, ge=1)
    socket_timeout: float = Field(default=5.0, gt=0)
    socket_connect_timeout: float = Field(default=5.0, gt=0)

    @property
    def url(self) -> str:
        if self.password:
            pwd = self.password.get_secret_value()
            return f"redis://:{pwd}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class KafkaSettings(BaseSettings):
    """Kafka streaming configuration."""

    model_config = SettingsConfigDict(
        env_prefix="KAFKA_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Comma-separated list of broker addresses",
    )
    topic_queries: str = Field(default="vectorlift.queries")
    topic_feedback: str = Field(default="vectorlift.feedback")
    consumer_group: str = Field(default="vectorlift-consumers")
    auto_offset_reset: str = Field(default="earliest", pattern="^(earliest|latest|none)$")
    max_poll_records: int = Field(default=500, ge=1)
    session_timeout_ms: int = Field(default=30_000, ge=1)

    @property
    def servers_list(self) -> list[str]:
        return [s.strip() for s in self.bootstrap_servers.split(",")]


class ModelSettings(BaseSettings):
    """ML model paths and hyperparameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    # Bi-encoder
    biencoder_model_path: str | None = Field(
        default=None,
        description="Path to fine-tuned bi-encoder (overrides pretrained)",
    )
    biencoder_pretrained: str = Field(
        default="sentence-transformers/msmarco-distilbert-base-tas-b",
        description="HuggingFace model ID for bi-encoder",
    )
    embedding_dim: int = Field(default=768, ge=64, le=4096)
    max_seq_length: int = Field(default=512, ge=32, le=4096)
    biencoder_batch_size: int = Field(default=64, ge=1)
    biencoder_device: str = Field(default="cpu", description="cpu | cuda | mps")

    # Cross-encoder reranker
    crossencoder_model_path: str | None = Field(
        default=None,
        description="Path to fine-tuned cross-encoder (overrides pretrained)",
    )
    crossencoder_pretrained: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace model ID for cross-encoder",
    )
    crossencoder_batch_size: int = Field(default=32, ge=1)
    crossencoder_max_length: int = Field(default=512, ge=32, le=4096)
    crossencoder_device: str = Field(default="cpu")

    @property
    def active_biencoder(self) -> str:
        """Returns the model to actually load — fine-tuned path wins."""
        return self.biencoder_model_path or self.biencoder_pretrained

    @property
    def active_crossencoder(self) -> str:
        return self.crossencoder_model_path or self.crossencoder_pretrained


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, ge=1, le=65535, alias="API_PORT")
    workers: int = Field(default=4, ge=1, alias="API_WORKERS")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        alias="CORS_ORIGINS",
    )
    cors_allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")
    request_timeout_seconds: int = Field(default=120, ge=1)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Root application settings.

    All sub-settings are instantiated lazily via properties so that they
    pick up the same .env file.  The top-level settings are read directly
    from environment / .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # Application
    app_env: AppEnv = Field(default=AppEnv.DEV)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    debug: bool = Field(default=False)
    secret_key: SecretStr = Field(default=SecretStr("change-me"))

    # External API keys
    openai_api_key: SecretStr | None = Field(default=None)
    hf_token: SecretStr | None = Field(default=None)

    # Dataset
    dataset_mode: DatasetMode = Field(default=DatasetMode.DEV)
    dataset_hf_name: str = Field(default="ms_marco")
    dataset_hf_config: str = Field(default="v1.1")
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    index_dir: Path = Field(default=Path("data/indexes"))

    # Retrieval defaults
    retrieval_default_mode: RetrievalMode = Field(default=RetrievalMode.HYBRID)
    top_k_default: Annotated[int, Field(ge=1, le=1000)] = 10
    rerank_top_n_default: Annotated[int, Field(ge=1, le=10_000)] = 100

    # Hybrid fusion
    hybrid_bm25_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3
    hybrid_dense_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    hybrid_fusion_strategy: HybridFusionStrategy = Field(
        default=HybridFusionStrategy.RECIPROCAL_RANK_FUSION
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF constant k")

    # Observability
    prometheus_port: int = Field(default=9090, ge=1, le=65535)
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    otel_service_name: str = Field(default="vectorlift-api")
    otel_traces_exporter: str = Field(default="otlp")
    sentry_dsn: str | None = Field(default=None)

    # Prefect
    prefect_api_url: AnyHttpUrl = Field(default="http://localhost:4200/api")  # type: ignore[assignment]

    # ---------------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_hybrid_weights(self) -> "Settings":
        total = round(self.hybrid_bm25_weight + self.hybrid_dense_weight, 6)
        if total != 1.0:
            raise ValueError(
                f"hybrid_bm25_weight + hybrid_dense_weight must equal 1.0, got {total}"
            )
        return self

    @model_validator(mode="after")
    def warn_insecure_secret(self) -> "Settings":
        if (
            self.app_env == AppEnv.PROD
            and self.secret_key.get_secret_value() == "change-me"
        ):
            raise ValueError("SECRET_KEY must be changed from the default in production.")
        return self

    # ---------------------------------------------------------------------------
    # Nested sub-settings (instantiated on first access)
    # ---------------------------------------------------------------------------

    @property
    def elasticsearch(self) -> ElasticsearchSettings:
        return ElasticsearchSettings()

    @property
    def qdrant(self) -> QdrantSettings:
        return QdrantSettings()

    @property
    def postgres(self) -> PostgresSettings:
        return PostgresSettings()

    @property
    def redis(self) -> RedisSettings:
        return RedisSettings()

    @property
    def kafka(self) -> KafkaSettings:
        return KafkaSettings()

    @property
    def model(self) -> ModelSettings:
        return ModelSettings()

    @property
    def api(self) -> APISettings:
        return APISettings()

    # ---------------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------------

    @property
    def is_dev(self) -> bool:
        return self.app_env == AppEnv.DEV

    @property
    def is_prod(self) -> bool:
        return self.app_env == AppEnv.PROD


# ---------------------------------------------------------------------------
# Cached singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application-wide settings singleton.

    The result is cached so the .env file is read only once per process.
    Call ``get_settings.cache_clear()`` in tests to reset.
    """
    return Settings()
