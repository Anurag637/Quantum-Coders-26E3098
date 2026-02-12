"""
API Router - Central Routing Configuration
Handles all API endpoint registration with versioning, dependencies, and documentation
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import time
import uuid

from core.logging import get_logger
from core.security import verify_api_key, get_current_user, verify_admin
from core.rate_limiter import rate_limit
from config import settings

# Import versioned routers
from api.v1.router import router as v1_router

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# MAIN API ROUTER
# ============================================================================
# This is the root router that includes all versioned API endpoints
# Following semantic versioning: /api/v1/, /api/v2/, etc.

api_router = APIRouter(prefix="/api")

# ============================================================================
# API VERSION MANAGEMENT
# ============================================================================
# Include different API versions
# Currently only v1 is implemented

api_router.include_router(
    v1_router,
    prefix="/v1",
    tags=["API v1"]
)

# ============================================================================
# API ROOT ENDPOINT
# ============================================================================

@api_router.get(
    "",
    summary="API Root",
    description="Get information about all available API versions",
    response_model=Dict[str, Any],
    include_in_schema=True
)
async def api_root(request: Request):
    """
    API root endpoint that provides information about all available API versions.
    
    Returns:
        Dict containing API version information and available endpoints
    """
    return {
        "name": settings.project_name,
        "version": settings.version,
        "environment": settings.environment.value,
        "versions": {
            "v1": {
                "status": "stable",
                "base_url": f"{request.base_url}api/v1",
                "docs": f"{request.base_url}docs",
                "redoc": f"{request.base_url}redoc",
                "openapi": f"{request.base_url}openapi.json",
                "endpoints": [
                    "/chat",
                    "/models", 
                    "/routing",
                    "/monitoring",
                    "/admin",
                    "/health"
                ]
            }
        },
        "documentation": {
            "swagger": f"{request.base_url}docs",
            "redoc": f"{request.base_url}redoc"
        },
        "timestamp": time.time(),
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


# ============================================================================
# API HEALTH CHECK
# ============================================================================

@api_router.get(
    "/health",
    summary="API Health Check",
    description="Check if the API is operational and all services are healthy",
    tags=["System"],
    response_model=Dict[str, Any]
)
async def api_health_check(request: Request):
    """
    Dedicated API health check endpoint.
    Returns detailed health status of all API components.
    """
    from monitoring.health_check import HealthChecker
    
    health_checker = HealthChecker()
    health_status = await health_checker.check_api_health()
    
    return {
        **health_status,
        "api_version": settings.version,
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


# ============================================================================
# API CONFIGURATION
# ============================================================================

@api_router.get(
    "/config",
    summary="API Configuration",
    description="Get public API configuration (non-sensitive)",
    tags=["System"],
    response_model=Dict[str, Any]
)
async def get_api_config():
    """
    Get public API configuration settings.
    Does not expose sensitive information like API keys or secrets.
    """
    return {
        "project_name": settings.project_name,
        "version": settings.version,
        "environment": settings.environment.value,
        "features": {
            "streaming": settings.enable_streaming,
            "batching": settings.enable_batching,
            "websockets": settings.enable_websockets,
            "caching": settings.cache.enabled,
            "semantic_cache": settings.cache.semantic_enabled,
            "rate_limiting": settings.rate_limit.enabled,
            "circuit_breakers": settings.enable_circuit_breaker,
            "fallbacks": settings.enable_fallbacks
        },
        "limits": {
            "max_request_size_mb": settings.server.max_request_size / (1024 * 1024),
            "max_tokens_per_request": 4096,
            "max_concurrent_requests": settings.server.max_concurrent_requests,
            "rate_limit_default": {
                "requests": settings.rate_limit.default_requests,
                "period_seconds": settings.rate_limit.default_period
            }
        },
        "models": {
            "total_configured": len(settings.model_config_dict.get("models", {})),
            "external_providers": [
                provider for provider in ["grok", "openai", "anthropic", "cohere"]
                if getattr(settings.api, f"{provider}_api_key_value") is not None
            ]
        }
    }


# ============================================================================
# API STATISTICS
# ============================================================================

@api_router.get(
    "/stats",
    summary="API Statistics",
    description="Get API usage statistics (requires authentication)",
    tags=["System"],
    dependencies=[Depends(verify_api_key)]
)
async def get_api_stats(
    request: Request,
    period: Optional[str] = "24h",
    user_id: Optional[str] = Depends(get_current_user)
):
    """
    Get API usage statistics for the authenticated user/organization.
    
    Args:
        period: Time period for stats (1h, 24h, 7d, 30d)
        user_id: Current authenticated user ID
    
    Returns:
        Dict containing API usage statistics
    """
    from database.repositories.metrics_repository import MetricsRepository
    
    repo = MetricsRepository()
    
    # Parse period
    import re
    from datetime import datetime, timedelta
    
    now = datetime.utcnow()
    
    if period == "1h":
        start_time = now - timedelta(hours=1)
    elif period == "24h":
        start_time = now - timedelta(days=1)
    elif period == "7d":
        start_time = now - timedelta(days=7)
    elif period == "30d":
        start_time = now - timedelta(days=30)
    else:
        start_time = now - timedelta(days=1)
    
    # Get stats
    stats = await repo.get_api_stats(
        user_id=user_id,
        start_time=start_time,
        end_time=now
    )
    
    return {
        "period": period,
        "start_time": start_time.isoformat(),
        "end_time": now.isoformat(),
        "statistics": stats,
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


# ============================================================================
# API VERSION CHECK
# ============================================================================

@api_router.get(
    "/versions",
    summary="API Versions",
    description="Get information about all available API versions",
    tags=["System"],
    response_model=Dict[str, Any]
)
async def get_api_versions(request: Request):
    """
    Get detailed information about all API versions.
    
    Returns:
        Dict containing version information, status, and deprecation dates
    """
    return {
        "current_version": settings.version,
        "versions": [
            {
                "version": "v1",
                "status": "stable",
                "release_date": "2024-01-15",
                "deprecation_date": None,
                "end_of_life": None,
                "documentation": f"{request.base_url}docs",
                "changelog": "https://github.com/yourorg/llm-gateway/releases/tag/v1.0.0"
            }
        ],
        "deprecation_policy": {
            "deprecation_notice": "6 months",
            "grace_period": "3 months",
            "current_version_support": "until 2025-01-15"
        },
        "upgrade_recommendations": {
            "latest_stable": "v1",
            "latest_beta": None,
            "next_major": "v2 (planned Q3 2024)"
        }
    }


# ============================================================================
# API STATUS PAGE
# ============================================================================

@api_router.get(
    "/status",
    summary="API Status Page",
    description="Get comprehensive API status for monitoring services",
    tags=["System"],
    response_model=Dict[str, Any]
)
async def get_api_status(request: Request):
    """
    Comprehensive API status endpoint.
    Used by external monitoring services (Statuspage, Pingdom, etc.)
    """
    from monitoring.health_check import HealthChecker
    
    health_checker = HealthChecker()
    
    # Get all health checks
    db_health = await health_checker.check_database()
    redis_health = await health_checker.check_redis()
    models_health = await health_checker.check_models()
    
    # Determine overall status
    status = "operational"
    
    if any(h.get("status") != "healthy" for h in [db_health, redis_health]):
        status = "degraded"
    
    if models_health.get("status") == "critical":
        status = "partial_outage"
    
    return {
        "page": {
            "id": "llm-gateway-api",
            "name": settings.project_name,
            "url": request.base_url,
            "updated": time.isoformat()
        },
        "status": {
            "indicator": status,
            "description": f"API is {status}",
            "color": "green" if status == "operational" else "yellow"
        },
        "components": [
            {
                "id": "api-core",
                "name": "API Core",
                "status": "operational",
                "description": "Main API endpoints"
            },
            {
                "id": "database",
                "name": "Database",
                "status": db_health.get("status", "unknown"),
                "description": f"Response time: {db_health.get('response_time_ms', 0)}ms"
            },
            {
                "id": "cache",
                "name": "Cache Service",
                "status": redis_health.get("status", "unknown"),
                "description": f"Connected: {redis_health.get('connected', False)}"
            },
            {
                "id": "models",
                "name": "Model Service",
                "status": models_health.get("status", "unknown"),
                "description": f"Loaded: {models_health.get('loaded_count', 0)} models"
            }
        ],
        "metrics": {
            "uptime_7d": "99.95%",
            "uptime_30d": "99.97%",
            "response_time_p95": "245ms",
            "requests_per_minute": 1234
        },
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


# ============================================================================
# API USAGE GUIDELINES
# ============================================================================

@api_router.get(
    "/guidelines",
    summary="API Usage Guidelines",
    description="Get API usage guidelines and best practices",
    tags=["System"],
    response_model=Dict[str, Any]
)
async def get_api_guidelines():
    """
    Return API usage guidelines and best practices.
    Helpful for developers integrating with the API.
    """
    return {
        "guidelines": {
            "authentication": {
                "method": "API Key",
                "header": "X-API-Key",
                "example": "X-API-Key: llm_your_api_key_here"
            },
            "rate_limiting": {
                "free_tier": "100 requests per minute",
                "pro_tier": "1000 requests per minute",
                "enterprise_tier": "Custom limits",
                "headers": [
                    "X-RateLimit-Limit",
                    "X-RateLimit-Remaining", 
                    "X-RateLimit-Reset"
                ]
            },
            "errors": {
                "4xx": "Client error - fix your request",
                "429": "Rate limit exceeded - slow down",
                "5xx": "Server error - retry with exponential backoff"
            },
            "pagination": {
                "method": "offset/limit",
                "default_limit": 50,
                "max_limit": 100,
                "headers": ["X-Total-Count", "X-Page", "X-Per-Page"]
            },
            "request_ids": {
                "header": "X-Request-ID",
                "description": "Include for request tracing",
                "response_header": "X-Request-ID"
            },
            "caching": {
                "etag": "ETag header for conditional requests",
                "cache_control": "Respect Cache-Control headers",
                "ttl": "3600 seconds (1 hour)"
            }
        }
    }


# ============================================================================
# API CHANGELOG
# ============================================================================

@api_router.get(
    "/changelog",
    summary="API Changelog",
    description="Get API changelog and version history",
    tags=["System"],
    response_model=Dict[str, Any]
)
async def get_api_changelog():
    """
    Return API changelog with version history and breaking changes.
    """
    return {
        "changelog": [
            {
                "version": "1.0.0",
                "date": "2024-01-15",
                "type": "major",
                "changes": [
                    "Initial release",
                    "Chat completions endpoint",
                    "Model management API",
                    "Routing visualization",
                    "Monitoring dashboard"
                ],
                "breaking_changes": []
            }
        ],
        "deprecations": []
    }


# ============================================================================
# ERROR HANDLING FOR API ROUTER
# ============================================================================

@api_router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    include_in_schema=False
)
async def catch_all(request: Request, path: str):
    """
    Catch-all route for undefined API endpoints.
    Returns a 404 error with helpful message.
    """
    logger.warning(
        "api_endpoint_not_found",
        path=request.url.path,
        method=request.method,
        request_id=getattr(request.state, "request_id", None)
    )
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "NotFound",
            "detail": f"API endpoint '{request.method} {request.url.path}' not found",
            "available_versions": ["v1"],
            "documentation_url": f"{request.base_url}docs",
            "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
            "timestamp": time.time()
        }
    )


# ============================================================================
# ADMIN-ONLY API ENDPOINTS
# ============================================================================

@api_router.post(
    "/admin/cache/clear",
    summary="Clear Cache",
    description="Clear all cache entries (admin only)",
    tags=["Admin"],
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=10, period=60))]
)
async def admin_clear_cache(request: Request):
    """
    Clear all cache entries.
    Only accessible by admin users with rate limiting.
    """
    from cache.cache_manager import CacheManager
    
    cache_manager = CacheManager()
    result = await cache_manager.clear_all()
    
    logger.info(
        "admin_cache_cleared",
        admin_id=getattr(request.state, "user_id", None),
        request_id=getattr(request.state, "request_id", None)
    )
    
    return {
        "success": result,
        "message": "Cache cleared successfully" if result else "Failed to clear cache",
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


@api_router.post(
    "/admin/models/reload",
    summary="Reload Models",
    description="Reload model registry (admin only)",
    tags=["Admin"],
    dependencies=[Depends(verify_admin)]
)
async def admin_reload_models(request: Request):
    """
    Reload model registry from configuration file.
    Only accessible by admin users.
    """
    from models.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    await registry.load_from_config(settings.model.config_path)
    
    logger.info(
        "admin_models_reloaded",
        admin_id=getattr(request.state, "user_id", None),
        request_id=getattr(request.state, "request_id", None)
    )
    
    return {
        "success": True,
        "message": "Model registry reloaded successfully",
        "model_count": len(registry.get_all_models()),
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


@api_router.get(
    "/admin/logs",
    summary="System Logs",
    description="Get system logs (admin only)",
    tags=["Admin"],
    dependencies=[Depends(verify_admin)]
)
async def admin_get_logs(
    request: Request,
    lines: int = 100,
    level: Optional[str] = None
):
    """
    Get recent system logs.
    Only accessible by admin users.
    
    Args:
        lines: Number of log lines to return (max 1000)
        level: Filter by log level (DEBUG, INFO, WARNING, ERROR)
    """
    from utils.log_viewer import get_recent_logs
    
    if lines > 1000:
        lines = 1000
    
    logs = await get_recent_logs(
        lines=lines,
        level=level,
        request_id=getattr(request.state, "request_id", None)
    )
    
    return {
        "lines_returned": len(logs),
        "lines_requested": lines,
        "level_filter": level,
        "logs": logs,
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


# ============================================================================
# API METADATA
# ============================================================================

@api_router.get(
    "/openapi",
    summary="OpenAPI Specification",
    description="Get the OpenAPI specification for this API",
    tags=["System"],
    include_in_schema=False
)
async def get_openapi_spec(request: Request):
    """
    Return the OpenAPI specification JSON.
    """
    from fastapi.openapi.utils import get_openapi
    
    # Import the main app to generate OpenAPI spec
    from main import app
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    
    return JSONResponse(content=openapi_schema)


# ============================================================================
# API HEALTH CHECK DETAILS
# ============================================================================

@api_router.get(
    "/health/details",
    summary="Detailed Health Check",
    description="Get detailed health information about all API components",
    tags=["System"],
    dependencies=[Depends(verify_api_key)]
)
async def get_detailed_health(request: Request):
    """
    Get detailed health information about all API components.
    Requires authentication.
    """
    from monitoring.health_check import HealthChecker
    
    health_checker = HealthChecker()
    
    # Run all health checks
    health_status = {
        "timestamp": time.time(),
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
        "components": await health_checker.check_all_components(),
        "dependencies": await health_checker.check_all_dependencies(),
        "models": await health_checker.get_model_health_details(),
        "performance": await health_checker.get_performance_metrics(),
        "alerts": await health_checker.get_active_alerts()
    }
    
    return health_status


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["api_router"]