from functools import lru_cache
from typing import List, Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from the project root (two levels above this file: app/core -> app -> backend -> project root)
_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===============================
    # Application
    # ===============================

    APP_NAME: str = "LLM Inference Gateway"
    APP_ENV: str = "development"
    DEBUG: bool = False
    VERSION: str = "0.1.0"

    API_V1_PREFIX: str = "/api/v1"

    # ===============================
    # Security
    # ===============================

    SECRET_KEY: str = Field(default="change-me-in-production", description="JWT secret key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    API_KEY_HEADER_NAME: str = "X-API-Key"

    # ===============================
    # Provider API Keys
    # ===============================

    # Optional: if unset, providers run in "mock" mode
    HUGGINGFACE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # ===============================
    # Database
    # ===============================

    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///", description="Async database URL (defaults to in-memory SQLite)")

    # ===============================
    # Redis
    # ===============================

    REDIS_URL: str = "redis://localhost:6379/0"

    # ===============================
    # Rate Limiting
    # ===============================

    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ===============================
    # CORS
    # ===============================

    CORS_ORIGINS: List[str] = ["*"]

    # ===============================
    # Model Settings
    # ===============================

    DEFAULT_MODEL: str = "mistral-7b-instruct"
    ENABLE_STREAMING: bool = True
    MAX_TOKENS_DEFAULT: int = 1024
    MAX_TOKENS_LIMIT: int = 8192
    MODEL_DIR: str = "models"


    # ===============================
    # Observability
    # ===============================

    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = False

    # ===============================
    # Logging
    # ===============================

    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance.
    """
    return Settings()

settings = get_settings()