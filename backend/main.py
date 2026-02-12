"""
LLM INFERENCE GATEWAY - MAIN APPLICATION ENTRY POINT
Production-ready FastAPI application with comprehensive middleware, error handling,
and lifecycle management for serving 15+ LLM models with intelligent routing.
"""

import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
import structlog
import psutil

from config import settings
from api.v1.router import api_router
from core.exceptions import LLMGatewayException
from core.logging import setup_logging, get_logger
from core.rate_limiter import RateLimiter
from core.circuit_breaker import CircuitBreakerRegistry
from gateway.gateway_handler import GatewayHandler
from models.model_manager import ModelManager
from models.model_registry import ModelRegistry
from cache.cache_manager import CacheManager
from cache.semantic_cache import SemanticCache
from monitoring.metrics import MetricsCollector
from monitoring.health_check import HealthChecker
from database.session import init_db, close_db_connections
from services.model_service import ModelService
from services.routing_service import RoutingService
from utils.version import get_version_info

# Setup structured logging
setup_logging()
logger = get_logger(__name__)

# ============================================================================
# GLOBAL INSTANCES
# ============================================================================
# These are initialized once and shared across all requests
# This pattern ensures efficient resource usage and consistent state

model_registry = ModelRegistry()
model_manager = ModelManager(model_registry)
cache_manager = CacheManager()
semantic_cache = SemanticCache()
rate_limiter = RateLimiter()
circuit_breaker_registry = CircuitBreakerRegistry()
metrics_collector = MetricsCollector()
health_checker = HealthChecker()
model_service = ModelService(model_manager, cache_manager)
routing_service = RoutingService(model_service)
gateway_handler = GatewayHandler(model_service, routing_service, cache_manager, metrics_collector)

# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - Handles startup and shutdown events
    
    Why this pattern?
    1. Clean startup/shutdown sequence
    2. Proper resource initialization and cleanup
    3. Graceful degradation on failures
    4. Comprehensive logging of lifecycle events
    5. Prevents memory leaks and orphaned connections
    """
    
    # ------------------------------------------------------------------------
    # STARTUP PHASE - Initialize all services
    # ------------------------------------------------------------------------
    startup_time = time.time()
    
    startup_msg = f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║     ██╗     ██╗███╗   ███╗     ██████╗  █████╗ ████████╗███████╗    ║
    ║     ██║     ██║████╗ ████║    ██╔════╝ ██╔══██╗╚══██╔══╝██╔════╝    ║
    ║     ██║     ██║██╔████╔██║    ██║  ███╗███████║   ██║   █████╗      ║
    ║     ██║     ██║██║╚██╔╝██║    ██║   ██║██╔══██║   ██║   ██╔══╝      ║
    ║     ███████╗██║██║ ╚═╝ ██║    ╚██████╔╝██║  ██║   ██║   ███████╗    ║
    ║     ╚══════╝╚═╝╚═╝     ╚═╝     ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝    ║
    ║                                                                      ║
    ║                    INFERENCE GATEWAY v{settings.version:<12}                    ║
    ║                                                                      ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  🚀 Starting LLM Inference Gateway...                               ║
    ║  📡 Environment: {settings.environment.value:<32} ║
    ║  🔧 Debug Mode: {str(settings.debug):<32} ║
    ║  🌐 Host: {settings.host:<32} ║
    ║  🚪 Port: {settings.port:<32} ║
    ║  📚 API Docs: http://{settings.host}:{settings.port}/docs           ║
    ║  📊 Metrics: http://{settings.host}:{settings.port}/metrics         ║
    ║  💓 Health: http://{settings.host}:{settings.port}/health           ║
    ║                                                                      ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  🤖 Loading Models...                                               ║
    """
    
    logger.info(startup_msg)
    
    try:
        # 1. Initialize Database Connection Pool
        logger.info("📦 Initializing database connection pool...")
        await init_db()
        logger.info("✅ Database initialized successfully")
        
        # 2. Initialize Redis Cache
        logger.info("🗄️ Initializing Redis cache...")
        await cache_manager.initialize()
        logger.info("✅ Cache initialized successfully")
        
        # 3. Load Model Registry
        logger.info("📋 Loading model registry...")
        await model_registry.load_from_config(settings.model_config_path)
        registry_stats = model_registry.get_stats()
        logger.info(f"✅ Model registry loaded: {registry_stats['total']} models configured")
        
        # 4. Load Pre-warmed Models
        if settings.environment != "production" or settings.prewarm_models:
            logger.info("🔥 Pre-warming frequently used models...")
            await model_manager.prewarm_default_models()
            logger.info("✅ Model pre-warming complete")
        
        # 5. Initialize Semantic Cache
        logger.info("🧠 Initializing semantic cache...")
        await semantic_cache.initialize()
        logger.info("✅ Semantic cache initialized")
        
        # 6. Initialize Circuit Breakers
        logger.info("🛡️ Initializing circuit breakers...")
        await circuit_breaker_registry.initialize()
        logger.info("✅ Circuit breakers initialized")
        
        # 7. Start Metrics Collection
        logger.info("📊 Starting metrics collection...")
        await metrics_collector.start()
        logger.info("✅ Metrics collection started")
        
        # 8. Start Health Checker
        logger.info("💓 Starting health checker...")
        await health_checker.start()
        logger.info("✅ Health checker started")
        
        # 9. Verify API Keys
        logger.info("🔑 Verifying API keys...")
        api_key_status = await verify_api_keys()
        logger.info(f"✅ API keys verified: {api_key_status}")
        
        # Calculate startup time
        startup_duration = time.time() - startup_time
        
        # ------------------------------------------------------------------------
        # STARTUP COMPLETE - Display success message
        # ------------------------------------------------------------------------
        success_msg = f"""
        ╠══════════════════════════════════════════════════════════════════════╣
        ║                                                                      ║
        ║  ✅ ALL SYSTEMS GO!                                                 ║
        ║                                                                      ║
        ║  📊 System Status:                                                  ║
        ║  ├─ 🤖 Models Configured: {registry_stats['total']:<30} ║
        ║  ├─ 🔥 Models Pre-warmed: {registry_stats['prewarmed']:<30} ║
        ║  ├─ 💾 Cache Size: {cache_manager.get_size():<30} ║
        ║  ├─ 🧠 Semantic Cache: {'Enabled' if settings.semantic_cache_enabled else 'Disabled':<29} ║
        ║  └─ 🛡️ Circuit Breakers: {circuit_breaker_registry.count():<30} ║
        ║                                                                      ║
        ║  ⚡ Performance Targets:                                             ║
        ║  ├─ 🚀 First Token: <100ms                                         ║
        ║  ├─ 📦 Cache Hit: <5ms                                             ║
        ║  ├─ 🎯 P95 Latency: <1000ms                                        ║
        ║  └─ 📈 Throughput: 100+ req/sec                                    ║
        ║                                                                      ║
        ║  🎉 Startup completed in {startup_duration:.2f} seconds!                     ║
        ║                                                                      ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        
        logger.info(success_msg)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}", exc_info=True)
        raise
    
    yield  # Application runs here
    
    # ------------------------------------------------------------------------
    # SHUTDOWN PHASE - Cleanup all resources
    # ------------------------------------------------------------------------
    
    shutdown_time = time.time()
    
    logger.info("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║  🛑 Shutting down LLM Inference Gateway...                          ║
    ║                                                                      ║
    ╠══════════════════════════════════════════════════════════════════════╣
    """)
    
    try:
        # 1. Unload all models
        logger.info("🔄 Unloading models...")
        await model_manager.unload_all_models()
        logger.info("✅ All models unloaded")
        
        # 2. Stop metrics collection
        logger.info("📊 Stopping metrics collection...")
        await metrics_collector.stop()
        logger.info("✅ Metrics collection stopped")
        
        # 3. Stop health checker
        logger.info("💓 Stopping health checker...")
        await health_checker.stop()
        logger.info("✅ Health checker stopped")
        
        # 4. Close cache connections
        logger.info("🗄️ Closing cache connections...")
        await cache_manager.close()
        logger.info("✅ Cache connections closed")
        
        # 5. Close database connections
        logger.info("📦 Closing database connections...")
        await close_db_connections()
        logger.info("✅ Database connections closed")
        
        shutdown_duration = time.time() - shutdown_time
        
        logger.info(f"""
        ╠══════════════════════════════════════════════════════════════════════╣
        ║                                                                      ║
        ║  ✅ Shutdown complete in {shutdown_duration:.2f} seconds!                    ║
        ║  👋 Goodbye!                                                        ║
        ║                                                                      ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """)
        
    except Exception as e:
        logger.error(f"❌ Shutdown failed: {str(e)}", exc_info=True)


# ============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="""
    # 🚀 LLM Inference Gateway and Prompt Router
    
    A **production-ready** LLM serving system with **intelligent routing**, 
    **semantic caching**, **load balancing**, and **comprehensive monitoring**.
    
    ## ✨ Key Features
    
    ### 🤖 Multi-Model Support
    - **15+ Models**: Local (Llama, Mistral, Falcon) + External (Grok, GPT-4, Claude, Cohere)
    - **Auto-scaling**: Dynamic model loading/unloading based on demand
    - **Quantization**: 4-bit/8-bit support for memory efficiency
    
    ### 🧠 Intelligent Routing
    - **Prompt Analysis**: Classifies prompts by type (code, reasoning, creative, Q&A)
    - **Cost Optimization**: Routes to cost-effective models while maintaining quality
    - **Latency Optimization**: Smart load balancing across backends
    - **Fallback Chains**: Automatic failover to backup models
    
    ### ⚡ Performance
    - **Semantic Caching**: 85% similarity threshold, sub-5ms cache hits
    - **Streaming**: First token in <100ms
    - **Batching**: Automatic request batching for high throughput
    - **P95 Latency**: <1000ms for 95th percentile
    
    ### 🛡️ Reliability
    - **Circuit Breakers**: Prevent cascading failures
    - **Rate Limiting**: Per-user and per-API key limits
    - **Retry Logic**: Exponential backoff with jitter
    - **Health Checks**: Automatic recovery from failures
    
    ### 📊 Observability
    - **Prometheus Metrics**: Request rates, latencies, error rates
    - **Grafana Dashboards**: Real-time visualization
    - **Structured Logging**: JSON logs with request IDs
    - **Distributed Tracing**: OpenTelemetry integration
    
    ## 🔑 Authentication
    
    All endpoints require authentication via:
    - **API Key**: `X-API-Key` header (recommended for applications)
    - **JWT Token**: `Authorization: Bearer <token>` (admin endpoints)
    
    ## 🎯 Quick Start
    
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/api/v1/chat/completions",
        headers={"X-API-Key": "your-api-key"},
        json={
            "messages": [
                {"role": "user", "content": "Write a Python function to reverse a string"}
            ],
            "model": "grok-beta",  # Optional: auto-routing if omitted
            "stream": False
        }
    )
    
    print(response.json()["choices"][0]["message"]["content"])
    ```
    
    ## 📚 Documentation
    
    - **Swagger UI**: `/docs`
    - **ReDoc**: `/redoc`
    - **OpenAPI JSON**: `/openapi.json`
    
    ## 🚦 Rate Limits
    
    | Tier | Requests/Minute | Tokens/Minute |
    |------|----------------|---------------|
    | Free | 100 | 10,000 |
    | Pro | 1,000 | 100,000 |
    | Enterprise | Custom | Custom |
    
    ## 🆘 Support
    
    - **Issues**: https://github.com/yourorg/llm-gateway/issues
    - **Documentation**: https://docs.llm-gateway.com
    - **Email**: support@llm-gateway.com
    
    ---
    **Version**: {settings.version} | **Environment**: {settings.environment.value}
    """,
    docs_url=None,  # Disable default docs to customize
    redoc_url=None,  # Disable default redoc to customize
    openapi_url="/openapi.json" if settings.environment != "production" else None,
    lifespan=lifespan,
    contact={
        "name": "LLM Gateway Team",
        "email": "team@llm-gateway.com",
        "url": "https://llm-gateway.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    terms_of_service="https://llm-gateway.com/terms",
)

# ============================================================================
# MIDDLEWARE STACK
# ============================================================================
# Order matters - each middleware wraps the next
# 1. Security headers
# 2. CORS
# 3. Trusted hosts
# 4. Request ID
# 5. Process time
# 6. Rate limiting
# 7. Authentication

# ----------------------------------------------------------------------------
# Security Headers Middleware
# ----------------------------------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # HSTS - Force HTTPS
    if settings.environment == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # XSS Protection
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    
    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Content Security Policy
    if settings.environment == "production":
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline';"
        )
    
    return response

# ----------------------------------------------------------------------------
# CORS Middleware
# ----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Request-ID",
        "X-Requested-With",
    ],
    expose_headers=[
        "X-Request-ID",
        "X-Process-Time",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    ],
    max_age=600,  # 10 minutes
)

# ----------------------------------------------------------------------------
# Trusted Host Middleware
# ----------------------------------------------------------------------------
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )

# ----------------------------------------------------------------------------
# Request ID Middleware
# ----------------------------------------------------------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    Add unique request ID to each request for tracing
    
    Why?
    1. Trace requests across logs
    2. Debug specific failures
    3. Correlate with external API calls
    4. Track user sessions
    """
    # Use client-provided ID or generate new one
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    
    request.state.request_id = request_id
    request.state.start_time = time.time()
    
    # Add to logger context
    structlog.contextvars.bind_contextvars(request_id=request_id)
    
    response = await call_next(request)
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    # Unbind context vars
    structlog.contextvars.unbind_contextvars("request_id")
    
    return response

# ----------------------------------------------------------------------------
# Process Time Middleware
# ----------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Track request processing time and log slow requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    process_time_ms = round(process_time * 1000, 2)
    
    # Add processing time to response headers
    response.headers["X-Process-Time-MS"] = str(process_time_ms)
    
    # Log slow requests (>1 second)
    if process_time > 1.0:
        logger.warning(
            "slow_request_detected",
            request_id=getattr(request.state, "request_id", None),
            method=request.method,
            path=request.url.path,
            duration_ms=process_time_ms,
            client_ip=request.client.host if request.client else None,
        )
    
    # Record metrics
    await metrics_collector.record_request(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
        duration=process_time,
        request_id=getattr(request.state, "request_id", None),
    )
    
    return response

# ----------------------------------------------------------------------------
# Rate Limiting Middleware
# ----------------------------------------------------------------------------
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting based on API key or IP address"""
    
    # Skip rate limiting for health checks and metrics
    if request.url.path in [
        "/health",
        "/health/ready",
        "/health/live",
        "/metrics",
        "/",
        "/favicon.ico",
    ]:
        return await call_next(request)
    
    # Get client identifier
    api_key = request.headers.get("X-API-Key")
    client_id = api_key or request.client.host
    
    # Apply rate limiting
    is_allowed, limit, remaining, reset_time = await rate_limiter.check_rate_limit(
        client_id,
        settings.rate_limit_requests,
        settings.rate_limit_period
    )
    
    if not is_allowed:
        logger.warning(
            "rate_limit_exceeded",
            client_id=client_id[:8] + "..." if len(client_id) > 8 else client_id,
            endpoint=request.url.path,
            request_id=getattr(request.state, "request_id", None),
        )
        
        response = JSONResponse(
            status_code=429,
            content={
                "error": "RateLimitExceeded",
                "detail": "Too many requests. Please try again later.",
                "limit": limit,
                "remaining": 0,
                "reset": reset_time,
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": time.time(),
            },
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(reset_time)),
                "Retry-After": str(int(reset_time - time.time())),
            },
        )
        
        # Record rate limit exceeded metric
        await metrics_collector.record_rate_limit_exceeded(
            client_id=client_id,
            endpoint=request.url.path,
        )
        
        return response
    
    response = await call_next(request)
    
    # Add rate limit headers to response
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(int(reset_time))
    
    return response

# ----------------------------------------------------------------------------
# Authentication Middleware
# ----------------------------------------------------------------------------
@app.middleware("http")
async def authentication_middleware(request: Request, call_next):
    """Validate authentication for protected endpoints"""
    
    # Public endpoints that don't require authentication
    public_paths = [
        "/",
        "/health",
        "/health/ready",
        "/health/live",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    ]
    
    if request.url.path in public_paths or request.url.path.startswith("/static"):
        return await call_next(request)
    
    # Check for API key in header
    api_key = request.headers.get("X-API-Key")
    
    if api_key:
        # Validate API key
        from core.security import verify_api_key
        user = await verify_api_key(api_key)
        if user:
            request.state.user_id = user["id"]
            request.state.is_admin = user.get("is_admin", False)
            request.state.rate_limit_quota = user.get("rate_limit_quota", 100)
            return await call_next(request)
    
    # Check for JWT token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        from core.security import verify_token
        user = await verify_token(token)
        if user:
            request.state.user_id = user["id"]
            request.state.is_admin = user.get("is_admin", False)
            return await call_next(request)
    
    # No valid authentication found
    logger.warning(
        "authentication_failed",
        path=request.url.path,
        client_ip=request.client.host if request.client else None,
        request_id=getattr(request.state, "request_id", None),
    )
    
    return JSONResponse(
        status_code=401,
        content={
            "error": "AuthenticationRequired",
            "detail": "Valid API key or JWT token required",
            "request_id": getattr(request.state, "request_id", None),
            "timestamp": time.time(),
        },
        headers={"WWW-Authenticate": "Bearer"},
    )

# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(LLMGatewayException)
async def llm_gateway_exception_handler(request: Request, exc: LLMGatewayException):
    """Handle custom LLM Gateway exceptions"""
    
    logger.error(
        "gateway_exception",
        error_type=exc.__class__.__name__,
        detail=exc.detail,
        status_code=exc.status_code,
        path=request.url.path,
        request_id=getattr(request.state, "request_id", None),
        exc_info=True,
    )
    
    # Record error metric
    await metrics_collector.record_error(
        error_type=exc.__class__.__name__,
        endpoint=request.url.path,
        status_code=exc.status_code,
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", None),
            "timestamp": time.time(),
        },
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    
    errors = []
    for error in exc.errors():
        errors.append({
            "loc": " -> ".join(str(loc) for loc in error["loc"]),
            "msg": error["msg"],
            "type": error["type"],
        })
    
    logger.warning(
        "validation_error",
        errors=errors,
        path=request.url.path,
        request_id=getattr(request.state, "request_id", None),
    )
    
    # Record error metric
    await metrics_collector.record_error(
        error_type="ValidationError",
        endpoint=request.url.path,
        status_code=422,
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "detail": errors,
            "request_id": getattr(request.state, "request_id", None),
            "timestamp": time.time(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    
    logger.warning(
        "http_exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        request_id=getattr(request.state, "request_id", None),
    )
    
    # Record error metric
    await metrics_collector.record_error(
        error_type=f"HTTP_{exc.status_code}",
        endpoint=request.url.path,
        status_code=exc.status_code,
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", None),
            "timestamp": time.time(),
        },
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all other unhandled exceptions"""
    
    error_id = str(uuid.uuid4())
    
    logger.error(
        "unhandled_exception",
        error_id=error_id,
        error_type=exc.__class__.__name__,
        error=str(exc),
        path=request.url.path,
        request_id=getattr(request.state, "request_id", None),
        exc_info=True,
    )
    
    # Record error metric
    await metrics_collector.record_error(
        error_type="InternalServerError",
        endpoint=request.url.path,
        status_code=500,
    )
    
    # In production, don't expose internal error details
    if settings.environment == "production":
        detail = "An unexpected error occurred. Our team has been notified."
    else:
        detail = str(exc)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": detail,
            "error_id": error_id,
            "request_id": getattr(request.state, "request_id", None),
            "timestamp": time.time(),
        },
    )

# ============================================================================
# CUSTOM DOCUMENTATION
# ============================================================================

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI documentation"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{settings.project_name} - API Documentation",
        swagger_favicon_url="/static/favicon.ico",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "displayRequestDuration": True,
            "filter": True,
            "tryItOutEnabled": True,
            "syntaxHighlight": {"theme": "monokai"},
        },
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Custom ReDoc documentation"""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=f"{settings.project_name} - ReDoc",
        redoc_favicon_url="/static/favicon.ico",
    )


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """Custom OpenAPI JSON"""
    return JSONResponse(get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    ))

# ============================================================================
# STATIC FILES
# ============================================================================

app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================================
# API ROUTES
# ============================================================================

# Include main API router
app.include_router(api_router, prefix=settings.api_v1_prefix)

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """
    Comprehensive health check endpoint
    
    Returns detailed health status of all system components:
    - Database connectivity
    - Redis cache
    - Model service status
    - Memory usage
    - Disk usage
    - Uptime
    """
    health_status = await health_checker.check_all()
    return health_status


@app.get("/health/ready", tags=["System"])
async def readiness_probe():
    """
    Readiness probe for Kubernetes
    
    Indicates if the service is ready to accept traffic
    """
    is_ready = await health_checker.is_ready()
    if is_ready:
        return {"status": "ready", "timestamp": time.time()}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "timestamp": time.time()}
        )


@app.get("/health/live", tags=["System"])
async def liveness_probe():
    """
    Liveness probe for Kubernetes
    
    Indicates if the service is alive
    """
    return {"status": "alive", "timestamp": time.time()}


@app.get("/health/startup", tags=["System"])
async def startup_probe():
    """
    Startup probe for Kubernetes
    
    Indicates if the service has completed startup
    """
    return {"status": "started", "timestamp": time.time()}

# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint
    
    Exposes metrics in Prometheus format for scraping:
    - HTTP request metrics (count, duration, size)
    - Model inference metrics (latency, tokens, errors)
    - Cache metrics (hit rate, size, evictions)
    - System metrics (CPU, memory, GPU)
    """
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint - API information and status
    """
    
    # Get system stats
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Get model stats
    model_stats = model_registry.get_stats()
    loaded_models = await model_manager.get_loaded_models_count()
    
    # Get cache stats
    cache_stats = await cache_manager.get_stats()
    
    return {
        "name": settings.project_name,
        "version": settings.version,
        "environment": settings.environment.value,
        "status": "operational",
        "timestamp": time.time(),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
        },
        "models": {
            "total_configured": model_stats["total"],
            "currently_loaded": loaded_models,
            "available": model_stats["available"],
            "external": model_stats["external"],
            "local": model_stats["local"],
        },
        "cache": {
            "enabled": settings.cache_enabled,
            "semantic_enabled": settings.semantic_cache_enabled,
            "hit_rate": cache_stats.get("hit_rate", 0),
            "size_mb": cache_stats.get("size_mb", 0),
        },
        "endpoints": {
            "chat": f"{settings.api_v1_prefix}/chat",
            "models": f"{settings.api_v1_prefix}/models",
            "routing": f"{settings.api_v1_prefix}/routing",
            "monitoring": f"{settings.api_v1_prefix}/monitoring",
            "admin": f"{settings.api_v1_prefix}/admin",
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "health": "/health",
        "metrics": "/metrics",
    }


# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/metrics")
async def websocket_metrics(websocket):
    """
    WebSocket endpoint for real-time metrics streaming
    
    Sends live metrics updates every second:
    - Request rate
    - Average latency
    - Error rate
    - Cache hit rate
    - Model status
    """
    await websocket.accept()
    
    logger.info(
        "websocket_connected",
        client=websocket.client.host if websocket.client else None,
    )
    
    try:
        while True:
            # Get live metrics
            metrics_data = await metrics_collector.get_live_metrics()
            
            # Send to client
            await websocket.send_json(metrics_data)
            
            # Wait 1 second before next update
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        
    finally:
        logger.info("websocket_disconnected")
        await websocket.close()


@app.websocket("/ws/logs")
async def websocket_logs(websocket):
    """
    WebSocket endpoint for real-time log streaming
    
    Only available for admin users
    """
    await websocket.accept()
    
    # Verify authentication
    api_key = websocket.headers.get("X-API-Key")
    if not api_key:
        await websocket.close(code=1008, reason="Authentication required")
        return
    
    user = await verify_api_key(api_key)
    if not user or not user.get("is_admin", False):
        await websocket.close(code=1008, reason="Admin access required")
        return
    
    logger.info(
        "websocket_logs_connected",
        user_id=user["id"],
        client=websocket.client.host if websocket.client else None,
    )
    
    try:
        # TODO: Implement log streaming
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"WebSocket logs error: {e}")
        
    finally:
        await websocket.close()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def verify_api_keys() -> Dict[str, bool]:
    """Verify all configured API keys are valid"""
    
    status = {
        "grok": False,
        "openai": False,
        "anthropic": False,
        "cohere": False,
        "huggingface": False,
    }
    
    # Check Grok API key
    if settings.grok_api_key:
        try:
            from backends.grok_backend import GrokBackend
            backend = GrokBackend({"api_key": settings.grok_api_key})
            status["grok"] = await backend.verify_api_key()
        except Exception as e:
            logger.warning(f"Grok API key verification failed: {e}")
    
    # Check OpenAI API key
    if settings.openai_api_key:
        try:
            from backends.openai_backend import OpenAIBackend
            backend = OpenAIBackend({"api_key": settings.openai_api_key})
            status["openai"] = await backend.verify_api_key()
        except Exception as e:
            logger.warning(f"OpenAI API key verification failed: {e}")
    
    # Check Anthropic API key
    if settings.anthropic_api_key:
        try:
            from backends.anthropic_backend import AnthropicBackend
            backend = AnthropicBackend({"api_key": settings.anthropic_api_key})
            status["anthropic"] = await backend.verify_api_key()
        except Exception as e:
            logger.warning(f"Anthropic API key verification failed: {e}")
    
    # Check Cohere API key
    if settings.cohere_api_key:
        try:
            from backends.cohere_backend import CohereBackend
            backend = CohereBackend({"api_key": settings.cohere_api_key})
            status["cohere"] = await backend.verify_api_key()
        except Exception as e:
            logger.warning(f"Cohere API key verification failed: {e}")
    
    # Check HuggingFace token
    if settings.huggingface_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=settings.huggingface_token)
            api.whoami()
            status["huggingface"] = True
        except Exception as e:
            logger.warning(f"HuggingFace token verification failed: {e}")
    
    return status


# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║     🚀 Starting development server...                               ║
    ║                                                                      ║
    ║     📡 http://{settings.host}:{settings.port}                              ║
    ║     📚 http://{settings.host}:{settings.port}/docs                       ║
    ║     📊 http://{settings.host}:{settings.port}/metrics                   ║
    ║     💓 http://{settings.host}:{settings.port}/health                    ║
    ║                                                                      ║
    ║     Press CTRL+C to stop                                           ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        reload_dirs=["backend"] if settings.debug else None,
        log_level="info",
        access_log=False,  # We use our own structured logging
        use_colors=True,
        timeout_keep_alive=30,
        limit_max_requests=None,
    )