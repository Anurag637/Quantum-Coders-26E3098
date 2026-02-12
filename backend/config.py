"""
Configuration Management - Production Ready
Centralized configuration with environment variable validation and type safety
Handles all settings for LLM Gateway including models, APIs, caching, and monitoring
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Set
from enum import Enum
from functools import lru_cache

from pydantic import (
    BaseSettings, 
    Field, 
    validator, 
    SecretStr, 
    PostgresDsn, 
    RedisDsn,
    AnyHttpUrl,
    BaseModel
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import yaml

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# ENUMS
# ============================================================================

class Environment(str, Enum):
    """Application environment enumeration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

    def is_production(self) -> bool:
        return self == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        return self == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        return self == Environment.TESTING


class LogLevel(str, Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelType(str, Enum):
    """Model type enumeration"""
    LLAMA = "llama"
    MISTRAL = "mistral"
    FALCON = "falcon"
    GPT = "gpt"
    CLAUDE = "claude"
    COHERE = "cohere"
    GROK = "grok"
    QWEN = "qwen"
    PHI = "phi"
    DEEPSEEK = "deepseek"
    STARCODER = "starcoder"
    EXTERNAL = "external"
    MOCK = "mock"


class CacheStrategy(str, Enum):
    """Cache strategy enumeration"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    SEMANTIC = "semantic"
    NONE = "none"


class RoutingStrategy(str, Enum):
    """Routing strategy enumeration"""
    LATENCY = "latency"
    COST = "cost"
    QUALITY = "quality"
    HYBRID = "hybrid"
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    ADAPTIVE = "adaptive"


# ============================================================================
# NESTED CONFIGURATION MODELS
# ============================================================================

class DatabaseConfig(BaseModel):
    """Database configuration"""
    url: PostgresDsn = Field(..., description="PostgreSQL connection URL")
    pool_size: int = Field(20, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(10, ge=0, le=50, description="Maximum overflow connections")
    pool_timeout: int = Field(30, ge=1, le=60, description="Pool timeout in seconds")
    pool_recycle: int = Field(3600, ge=60, le=86400, description="Connection recycle time")
    echo: bool = Field(False, description="Echo SQL queries")
    retry_limit: int = Field(3, ge=0, le=10, description="Connection retry limit")
    retry_interval: int = Field(1, ge=1, le=10, description="Retry interval in seconds")


class RedisConfig(BaseModel):
    """Redis cache configuration"""
    url: RedisDsn = Field(..., description="Redis connection URL")
    max_connections: int = Field(50, ge=1, le=200, description="Maximum connections")
    socket_timeout: int = Field(5, ge=1, le=30, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(5, ge=1, le=30, description="Connect timeout")
    retry_on_timeout: bool = Field(True, description="Retry on timeout")
    ssl: bool = Field(False, description="Use SSL")
    decode_responses: bool = Field(True, description="Decode responses")
    health_check_interval: int = Field(30, ge=5, le=120, description="Health check interval")


class CacheConfig(BaseModel):
    """Cache configuration"""
    enabled: bool = Field(True, description="Enable caching")
    strategy: CacheStrategy = Field(CacheStrategy.TTL, description="Cache strategy")
    default_ttl: int = Field(3600, ge=60, le=86400, description="Default TTL in seconds")
    max_size: int = Field(10000, ge=100, le=1000000, description="Maximum cache entries")
    
    # Semantic cache settings
    semantic_enabled: bool = Field(True, description="Enable semantic caching")
    similarity_threshold: float = Field(0.85, ge=0.5, le=1.0, description="Semantic similarity threshold")
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Embedding model for semantic cache")
    max_embeddings_cache: int = Field(5000, ge=100, le=50000, description="Max embeddings in cache")
    
    # Redis cache settings
    redis: RedisConfig


class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""
    enabled: bool = Field(True, description="Enable rate limiting")
    default_requests: int = Field(100, ge=1, le=10000, description="Default requests per period")
    default_period: int = Field(60, ge=1, le=3600, description="Default period in seconds")
    
    # Per-tier limits
    free_requests: int = Field(100, ge=1, le=1000, description="Free tier requests per minute")
    pro_requests: int = Field(1000, ge=100, le=10000, description="Pro tier requests per minute")
    enterprise_requests: int = Field(10000, ge=1000, le=100000, description="Enterprise tier requests per minute")
    
    # Burst limits
    burst_multiplier: float = Field(2.0, ge=1.0, le=5.0, description="Burst multiplier")
    burst_duration: int = Field(10, ge=1, le=60, description="Burst duration in seconds")


class ModelConfig(BaseModel):
    """Model configuration"""
    models_dir: Path = Field(Path("/app/models"), description="Models directory")
    config_path: Path = Field(Path("/app/config/models.yaml"), description="Model config file path")
    routing_config_path: Path = Field(Path("/app/config/routing_rules.yaml"), description="Routing rules path")
    
    # Loading settings
    prewarm_models: bool = Field(True, description="Pre-warm models on startup")
    max_concurrent_loads: int = Field(3, ge=1, le=10, description="Max concurrent model loads")
    load_timeout: int = Field(300, ge=60, le=3600, description="Model load timeout in seconds")
    unload_after_idle: int = Field(3600, ge=300, le=86400, description="Unload idle models after seconds")
    
    # Quantization
    enable_quantization: bool = Field(True, description="Enable model quantization")
    default_quantization: str = Field("4bit", description="Default quantization bits")
    
    # GPU settings
    gpu_enabled: bool = Field(True, description="Enable GPU acceleration")
    gpu_memory_fraction: float = Field(0.9, ge=0.1, le=1.0, description="GPU memory fraction to use")
    cuda_visible_devices: str = Field("0", description="CUDA visible devices")
    
    # Fallback settings
    enable_fallbacks: bool = Field(True, description="Enable model fallbacks")
    max_retries: int = Field(3, ge=0, le=10, description="Max retries on failure")
    retry_delay: int = Field(1, ge=0, le=10, description="Retry delay in seconds")


class APIConfig(BaseModel):
    """External API configuration"""
    # Grok API (xAI)
    grok_api_key: Optional[SecretStr] = Field(None, description="Grok API key")
    grok_api_url: AnyHttpUrl = Field("https://api.x.ai/v1", description="Grok API URL")
    grok_timeout: int = Field(60, ge=1, le=300, description="Grok API timeout")
    grok_max_retries: int = Field(3, ge=0, le=10, description="Grok max retries")
    
    # OpenAI API
    openai_api_key: Optional[SecretStr] = Field(None, description="OpenAI API key")
    openai_api_url: AnyHttpUrl = Field("https://api.openai.com/v1", description="OpenAI API URL")
    openai_organization: Optional[str] = Field(None, description="OpenAI organization ID")
    openai_timeout: int = Field(60, ge=1, le=300, description="OpenAI timeout")
    
    # Anthropic API
    anthropic_api_key: Optional[SecretStr] = Field(None, description="Anthropic API key")
    anthropic_api_url: AnyHttpUrl = Field("https://api.anthropic.com/v1", description="Anthropic API URL")
    anthropic_version: str = Field("2023-06-01", description="Anthropic API version")
    anthropic_timeout: int = Field(60, ge=1, le=300, description="Anthropic timeout")
    
    # Cohere API
    cohere_api_key: Optional[SecretStr] = Field(None, description="Cohere API key")
    cohere_api_url: AnyHttpUrl = Field("https://api.cohere.ai/v1", description="Cohere API URL")
    cohere_timeout: int = Field(60, ge=1, le=300, description="Cohere timeout")
    
    # HuggingFace
    huggingface_token: Optional[SecretStr] = Field(None, description="HuggingFace token")
    huggingface_cache_dir: Path = Field(Path("/root/.cache/huggingface"), description="HuggingFace cache dir")
    huggingface_offline: bool = Field(False, description="HuggingFace offline mode")
    
    # General API settings
    global_timeout: int = Field(30, ge=1, le=300, description="Global API timeout")
    global_max_retries: int = Field(3, ge=0, le=10, description="Global max retries")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")
    proxy: Optional[str] = Field(None, description="HTTP proxy URL")


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    enabled: bool = Field(True, description="Enable metrics collection")
    metrics_port: int = Field(9090, ge=1024, le=65535, description="Metrics port")
    prometheus_retention: str = Field("15d", description="Prometheus retention period")
    
    # Grafana
    grafana_password: Optional[SecretStr] = Field(None, description="Grafana admin password")
    grafana_port: int = Field(3001, ge=1024, le=65535, description="Grafana port")
    
    # Tracing
    tracing_enabled: bool = Field(False, description="Enable distributed tracing")
    jaeger_host: Optional[str] = Field(None, description="Jaeger agent host")
    jaeger_port: int = Field(6831, ge=1024, le=65535, description="Jaeger agent port")
    
    # Alerts
    alerting_enabled: bool = Field(True, description="Enable alerting")
    slack_webhook_url: Optional[AnyHttpUrl] = Field(None, description="Slack webhook URL")
    email_alerts: List[str] = Field(default_factory=list, description="Alert email recipients")


class SecurityConfig(BaseModel):
    """Security configuration"""
    secret_key: SecretStr = Field(..., description="JWT secret key")
    algorithm: str = Field("HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(30, ge=5, le=1440, description="Access token expiry")
    refresh_token_expire_days: int = Field(7, ge=1, le=30, description="Refresh token expiry")
    
    # API Keys
    api_key_header: str = Field("X-API-Key", description="API key header name")
    api_key_prefix: str = Field("llm_", description="API key prefix")
    api_key_length: int = Field(32, ge=16, le=64, description="API key length")
    
    # CORS
    cors_origins: List[AnyHttpUrl] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:3001"],
        description="CORS allowed origins"
    )
    cors_allow_credentials: bool = Field(True, description="CORS allow credentials")
    cors_allow_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        description="CORS allowed methods"
    )
    cors_allow_headers: List[str] = Field(
        default_factory=lambda: ["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
        description="CORS allowed headers"
    )
    
    # SSL
    ssl_enabled: bool = Field(False, description="Enable SSL")
    ssl_cert_path: Optional[Path] = Field(None, description="SSL certificate path")
    ssl_key_path: Optional[Path] = Field(None, description="SSL key path")
    
    # HSTS
    hsts_enabled: bool = Field(True, description="Enable HSTS")
    hsts_max_age: int = Field(31536000, ge=0, description="HSTS max age in seconds")


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: LogLevel = Field(LogLevel.INFO, description="Log level")
    format: str = Field("json", description="Log format (json or console)")
    file_path: Optional[Path] = Field(None, description="Log file path")
    max_size_mb: int = Field(100, ge=10, le=1000, description="Max log file size in MB")
    backup_count: int = Field(5, ge=1, le=30, description="Number of backup files")
    
    # Structured logging
    json_indent: Optional[int] = Field(None, description="JSON indent for pretty printing")
    include_timestamps: bool = Field(True, description="Include timestamps")
    include_level: bool = Field(True, description="Include log level")
    include_traceback: bool = Field(True, description="Include traceback in errors")
    
    # Log filtering
    exclude_paths: List[str] = Field(
        default_factory=lambda: ["/health", "/metrics", "/"],
        description="Paths to exclude from logging"
    )
    exclude_status_codes: List[int] = Field(
        default_factory=lambda: [200, 304],
        description="Status codes to exclude from logging"
    )


class ServerConfig(BaseModel):
    """Server configuration"""
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, ge=1024, le=65535, description="Server port")
    workers: int = Field(4, ge=1, le=32, description="Number of worker processes")
    
    # Timeouts
    keep_alive_timeout: int = Field(30, ge=5, le=120, description="Keep alive timeout")
    graceful_timeout: int = Field(60, ge=10, le=300, description="Graceful shutdown timeout")
    
    # Rate limiting
    max_request_size: int = Field(1024 * 1024 * 10, description="Max request size in bytes")  # 10MB
    max_concurrent_requests: int = Field(100, ge=10, le=1000, description="Max concurrent requests")
    
    # Headers
    proxy_headers: bool = Field(True, description="Trust proxy headers")
    forwarded_allow_ips: str = Field("*", description="Forwarded allow IPs")


# ============================================================================
# MAIN SETTINGS CLASS
# ============================================================================

class Settings(BaseSettings):
    """
    Main application settings - Single source of truth for all configuration
    
    This class loads configuration from:
    1. Environment variables
    2. .env file
    3. Default values
    4. YAML config files (models, routing)
    
    All settings are validated and type-checked at startup
    """
    
    # ========================================================================
    # APPLICATION SETTINGS
    # ========================================================================
    
    # Basic info
    project_name: str = Field(
        "LLM Inference Gateway",
        description="Project name"
    )
    version: str = Field(
        "1.0.0",
        description="Application version"
    )
    description: str = Field(
        "Production-ready LLM serving system with intelligent routing and caching",
        description="Application description"
    )
    environment: Environment = Field(
        Environment.DEVELOPMENT,
        description="Application environment"
    )
    debug: bool = Field(
        False,
        description="Debug mode"
    )
    
    # ========================================================================
    # NESTED CONFIGURATIONS
    # ========================================================================
    
    # Server configuration
    server: ServerConfig = ServerConfig()
    
    # Database configuration
    database: DatabaseConfig = Field(
        default_factory=lambda: DatabaseConfig(
            url=os.getenv("DATABASE_URL", "postgresql://llm_admin:admin123@postgres:5432/llm_gateway")
        )
    )
    
    # Cache configuration
    cache: CacheConfig = Field(
        default_factory=lambda: CacheConfig(
            redis=RedisConfig(url=os.getenv("REDIS_URL", "redis://redis:6379/0"))
        )
    )
    
    # Rate limiting configuration
    rate_limit: RateLimitConfig = RateLimitConfig()
    
    # Model configuration
    model: ModelConfig = ModelConfig()
    
    # External API configuration
    api: APIConfig = APIConfig()
    
    # Monitoring configuration
    monitoring: MonitoringConfig = MonitoringConfig()
    
    # Security configuration
    security: SecurityConfig = Field(
        default_factory=lambda: SecurityConfig(
            secret_key=os.getenv("SECRET_KEY", "development-secret-key-change-in-production")
        )
    )
    
    # Logging configuration
    logging: LoggingConfig = LoggingConfig()
    
    # ========================================================================
    # FEATURE FLAGS
    # ========================================================================
    
    # Core features
    enable_api: bool = Field(True, description="Enable API endpoints")
    enable_streaming: bool = Field(True, description="Enable streaming responses")
    enable_batching: bool = Field(True, description="Enable request batching")
    enable_websockets: bool = Field(True, description="Enable WebSocket endpoints")
    
    # Advanced features
    enable_circuit_breaker: bool = Field(True, description="Enable circuit breakers")
    enable_retries: bool = Field(True, description="Enable retry logic")
    enable_fallbacks: bool = Field(True, description="Enable model fallbacks")
    enable_load_balancing: bool = Field(True, description="Enable load balancing")
    
    # Experimental features
    enable_experimental_features: bool = Field(False, description="Enable experimental features")
    enable_quantization: bool = Field(True, description="Enable model quantization")
    enable_semantic_cache: bool = Field(True, description="Enable semantic caching")
    
    # ========================================================================
    # ROUTING SETTINGS
    # ========================================================================
    
    default_routing_strategy: RoutingStrategy = Field(
        RoutingStrategy.HYBRID,
        description="Default routing strategy"
    )
    
    routing_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "latency": 0.4,
            "cost": 0.3,
            "quality": 0.3
        },
        description="Routing strategy weights"
    )
    
    enable_routing_analytics: bool = Field(
        True,
        description="Enable routing decision analytics"
    )
    
    # ========================================================================
    # VALIDATORS
    # ========================================================================
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        """Validate and normalize environment"""
        if isinstance(v, str):
            v = v.lower()
            if v == "prod":
                v = "production"
            elif v == "dev":
                v = "development"
        return v
    
    @validator("debug")
    def set_debug_from_environment(cls, v, values):
        """Auto-set debug mode based on environment"""
        if "environment" in values:
            return v or values["environment"].is_development()
        return v
    
    @validator("routing_weights")
    def validate_routing_weights(cls, v):
        """Validate routing weights sum to 1.0"""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Routing weights must sum to 1.0, got {total}")
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # ========================================================================
    # MODEL CONFIG
    # ========================================================================
    
    @property
    def model_config_dict(self) -> Dict[str, Any]:
        """Load and cache model configuration from YAML"""
        if not hasattr(self, "_model_config"):
            try:
                path = self.model.config_path
                if path.exists():
                    with open(path, "r") as f:
                        self._model_config = yaml.safe_load(f)
                else:
                    self._model_config = {"models": {}}
            except Exception as e:
                self._model_config = {"models": {}}
        return self._model_config
    
    @property
    def routing_config_dict(self) -> Dict[str, Any]:
        """Load and cache routing configuration from YAML"""
        if not hasattr(self, "_routing_config"):
            try:
                path = self.model.routing_config_path
                if path.exists():
                    with open(path, "r") as f:
                        self._routing_config = yaml.safe_load(f)
                else:
                    self._routing_config = {"rules": [], "strategies": {}}
            except Exception as e:
                self._routing_config = {"rules": [], "strategies": {}}
        return self._routing_config
    
    # ========================================================================
    # API KEY PROPERTIES
    # ========================================================================
    
    @property
    def grok_api_key_value(self) -> Optional[str]:
        """Get Grok API key value"""
        return self.api.grok_api_key.get_secret_value() if self.api.grok_api_key else None
    
    @property
    def openai_api_key_value(self) -> Optional[str]:
        """Get OpenAI API key value"""
        return self.api.openai_api_key.get_secret_value() if self.api.openai_api_key else None
    
    @property
    def anthropic_api_key_value(self) -> Optional[str]:
        """Get Anthropic API key value"""
        return self.api.anthropic_api_key.get_secret_value() if self.api.anthropic_api_key else None
    
    @property
    def cohere_api_key_value(self) -> Optional[str]:
        """Get Cohere API key value"""
        return self.api.cohere_api_key.get_secret_value() if self.api.cohere_api_key else None
    
    @property
    def huggingface_token_value(self) -> Optional[str]:
        """Get HuggingFace token value"""
        return self.api.huggingface_token.get_secret_value() if self.api.huggingface_token else None
    
    # ========================================================================
    # SECURITY PROPERTIES
    # ========================================================================
    
    @property
    def secret_key_value(self) -> str:
        """Get secret key value"""
        return self.security.secret_key.get_secret_value()
    
    @property
    def grafana_password_value(self) -> Optional[str]:
        """Get Grafana password value"""
        return self.monitoring.grafana_password.get_secret_value() if self.monitoring.grafana_password else None
    
    # ========================================================================
    # URL PROPERTIES
    # ========================================================================
    
    @property
    def api_url(self) -> str:
        """Get API URL"""
        protocol = "https" if self.security.ssl_enabled else "http"
        return f"{protocol}://{self.server.host}:{self.server.port}"
    
    @property
    def docs_url(self) -> str:
        """Get docs URL"""
        return f"{self.api_url}/docs" if not self.environment.is_production() else None
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate all configured API keys are present"""
        status = {}
        
        # Check if external APIs are configured in model config
        for model_id, model_config in self.model_config_dict.get("models", {}).items():
            if model_config.get("type") == "external":
                provider = model_config.get("provider", "").lower()
                
                if "xai" in provider or "grok" in provider:
                    status["grok"] = self.grok_api_key_value is not None
                elif "openai" in provider:
                    status["openai"] = self.openai_api_key_value is not None
                elif "anthropic" in provider:
                    status["anthropic"] = self.anthropic_api_key_value is not None
                elif "cohere" in provider:
                    status["cohere"] = self.cohere_api_key_value is not None
        
        return status
    
    # ========================================================================
    # MODEL CONFIG
    # ========================================================================
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Why LRU cache?
    1. Settings are read-only after initialization
    2. Expensive to validate on every request
    3. Ensures consistent configuration across the application
    4. Reduces memory usage
    5. Thread-safe
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Global settings instance - use this throughout the application
settings = get_settings()


# ============================================================================
# ENVIRONMENT-SPECIFIC OVERRIDES
# ============================================================================

def configure_production_settings():
    """Apply production-specific overrides"""
    if settings.environment.is_production():
        settings.debug = False
        settings.logging.level = LogLevel.INFO
        settings.security.hsts_enabled = True
        settings.server.proxy_headers = True
        settings.monitoring.enabled = True
        settings.cache.enabled = True
        settings.rate_limit.enabled = True


def configure_development_settings():
    """Apply development-specific overrides"""
    if settings.environment.is_development():
        settings.debug = True
        settings.logging.level = LogLevel.DEBUG
        settings.logging.format = "console"
        settings.logging.json_indent = 2
        settings.security.hsts_enabled = False
        settings.database.echo = True
        settings.model.prewarm_models = False  # Don't prewarm in dev to save resources


def configure_testing_settings():
    """Apply testing-specific overrides"""
    if settings.environment.is_testing():
        settings.debug = True
        settings.logging.level = LogLevel.WARNING
        settings.database.url = "postgresql://llm_admin:admin123@localhost:5432/llm_gateway_test"
        settings.cache.redis.url = "redis://localhost:6379/1"
        settings.rate_limit.enabled = False
        settings.model.prewarm_models = False
        settings.enable_experimental_features = False


# Apply environment-specific configurations
configure_production_settings()
configure_development_settings()
configure_testing_settings()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Settings",
    "settings",
    "Environment",
    "LogLevel",
    "ModelType",
    "CacheStrategy",
    "RoutingStrategy",
    "DatabaseConfig",
    "RedisConfig",
    "CacheConfig",
    "RateLimitConfig",
    "ModelConfig",
    "APIConfig",
    "MonitoringConfig",
    "SecurityConfig",
    "LoggingConfig",
    "ServerConfig",
]