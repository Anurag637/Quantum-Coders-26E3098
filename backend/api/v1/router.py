"""
API v1 Router - Version 1 API Endpoint Registration
Implements the core LLM Gateway API endpoints with version-specific routing
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any, List, Optional
import time
import uuid

from core.logging import get_logger
from core.security import verify_api_key, get_current_user, verify_admin
from core.rate_limiter import rate_limit
from config import settings

# Import all v1 endpoint routers
from api.v1.endpoints import (
    chat,
    models,
    routing,
    monitoring,
    admin,
    health,
    embeddings,
    completions,
    files,
    batches
)

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# V1 API ROUTER
# ============================================================================
# This router aggregates all v1 API endpoints
# Version: 1.0.0
# Stability: Stable
# Deprecation Date: None
# ============================================================================

router = APIRouter(prefix="/v1")

# ============================================================================
# API VERSION METADATA
# ============================================================================

API_V1_METADATA = {
    "version": "1.0.0",
    "release_date": "2024-01-15",
    "stability": "stable",
    "deprecation_date": None,
    "end_of_life": None,
    "changelog": "/api/v1/changelog",
    "documentation": "/docs",
    "openapi": "/openapi.json"
}

# ============================================================================
# REGISTER ALL V1 ENDPOINTS
# ============================================================================

# ----------------------------------------------------------------------------
# Chat & Completions Endpoints
# ----------------------------------------------------------------------------
# Core LLM inference endpoints
# Rate limit: 100 requests per minute (default)
# Authentication: Required
# ----------------------------------------------------------------------------

router.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit())]
)

router.include_router(
    completions.router,
    prefix="/completions",
    tags=["Completions"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit())]
)

# ----------------------------------------------------------------------------
# Model Management Endpoints
# ----------------------------------------------------------------------------
# Model listing, loading, unloading, and metadata
# Rate limit: 30 requests per minute
# Authentication: Required
# ----------------------------------------------------------------------------

router.include_router(
    models.router,
    prefix="/models",
    tags=["Models"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit(limit=30, period=60))]
)

# ----------------------------------------------------------------------------
# Routing & Optimization Endpoints
# ----------------------------------------------------------------------------
# Intelligent prompt routing, load balancing, cost optimization
# Rate limit: 30 requests per minute
# Authentication: Required
# ----------------------------------------------------------------------------

router.include_router(
    routing.router,
    prefix="/routing",
    tags=["Routing"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit(limit=30, period=60))]
)

# ----------------------------------------------------------------------------
# Monitoring & Observability Endpoints
# ----------------------------------------------------------------------------
# Metrics, logs, traces, and health checks
# Rate limit: 60 requests per minute
# Authentication: Required (some endpoints public)
# ----------------------------------------------------------------------------

router.include_router(
    monitoring.router,
    prefix="/monitoring",
    tags=["Monitoring"],
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)

# ----------------------------------------------------------------------------
# Admin & Configuration Endpoints
# ----------------------------------------------------------------------------
# System configuration, cache management, user administration
# Rate limit: 10 requests per minute
# Authentication: Admin only
# ----------------------------------------------------------------------------

router.include_router(
    admin.router,
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=10, period=60))]
)

# ----------------------------------------------------------------------------
# System Health Endpoints
# ----------------------------------------------------------------------------
# Public health check endpoints (no authentication required)
# Rate limit: Higher limits for health checks
# ----------------------------------------------------------------------------

router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
    dependencies=[Depends(rate_limit(limit=200, period=60))]
)

# ----------------------------------------------------------------------------
# Embeddings Endpoints
# ----------------------------------------------------------------------------
# Text embeddings generation for semantic search and similarity
# Rate limit: 100 requests per minute
# Authentication: Required
# ----------------------------------------------------------------------------

router.include_router(
    embeddings.router,
    prefix="/embeddings",
    tags=["Embeddings"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit())]
)

# ----------------------------------------------------------------------------
# File Processing Endpoints
# ----------------------------------------------------------------------------
# Upload, process, and analyze files (PDF, DOCX, PPTX, etc.)
# Rate limit: 10 requests per minute
# Authentication: Required
# ----------------------------------------------------------------------------

router.include_router(
    files.router,
    prefix="/files",
    tags=["Files"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit(limit=10, period=60))]
)

# ----------------------------------------------------------------------------
# Batch Processing Endpoints
# ----------------------------------------------------------------------------
# Asynchronous batch jobs for large-scale processing
# Rate limit: 5 requests per minute
# Authentication: Required
# ----------------------------------------------------------------------------

router.include_router(
    batches.router,
    prefix="/batches",
    tags=["Batches"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit(limit=5, period=60))]
)

# ============================================================================
# V1 API ROOT ENDPOINT
# ============================================================================

@router.get(
    "",
    summary="API v1 Root",
    description="Get information about the v1 API version and available endpoints",
    response_model=Dict[str, Any],
    include_in_schema=True
)
async def v1_api_root(request: Request):
    """
    API v1 root endpoint that provides version information and available endpoints.
    
    Returns:
        Dict containing v1 API version information and available endpoints
    """
    # Build endpoint list dynamically from registered routers
    endpoints = []
    for route in router.routes:
        if hasattr(route, "path") and route.path.startswith("/v1"):
            # Skip the root endpoint itself
            if route.path != "/v1":
                endpoints.append({
                    "path": route.path.replace("/v1", ""),
                    "name": route.name,
                    "methods": list(route.methods) if hasattr(route, "methods") else ["GET"]
                })
    
    return {
        "version": API_V1_METADATA["version"],
        "release_date": API_V1_METADATA["release_date"],
        "stability": API_V1_METADATA["stability"],
        "deprecation_date": API_V1_METADATA["deprecation_date"],
        "end_of_life": API_V1_METADATA["end_of_life"],
        "base_url": f"{request.base_url}api/v1",
        "documentation": f"{request.base_url}docs",
        "openapi": f"{request.base_url}openapi.json",
        "changelog": f"{request.base_url}api/v1/changelog",
        "endpoints": endpoints,
        "features": {
            "chat_completions": True,
            "streaming": settings.enable_streaming,
            "embeddings": True,
            "file_processing": True,
            "batch_processing": True,
            "semantic_caching": settings.cache.semantic_enabled,
            "intelligent_routing": settings.enable_circuit_breaker
        },
        "limits": {
            "max_prompt_length": 4096,
            "max_tokens_per_request": 4096,
            "max_file_size_mb": 25,
            "max_batch_size": 100
        },
        "timestamp": time.time(),
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


# ============================================================================
# V1 API CHANGELOG
# ============================================================================

@router.get(
    "/changelog",
    summary="API v1 Changelog",
    description="Get the changelog for API v1 version history",
    response_model=Dict[str, Any],
    include_in_schema=True
)
async def v1_changelog():
    """
    Return the changelog for API v1 with all version updates.
    """
    return {
        "version": "1.0.0",
        "changelog": [
            {
                "version": "1.0.0",
                "date": "2024-01-15",
                "type": "major",
                "changes": [
                    "Initial release of API v1",
                    "Chat completions endpoint with streaming support",
                    "Model management API with 15+ models",
                    "Intelligent prompt routing with 4 strategies",
                    "Semantic caching with 85% similarity threshold",
                    "Real-time monitoring dashboard",
                    "Admin API for system management",
                    "File processing for PDF, DOCX, PPTX",
                    "Batch processing for large-scale jobs",
                    "Embeddings generation for semantic search"
                ],
                "breaking_changes": [],
                "deprecations": []
            }
        ],
        "upcoming_changes": [
            {
                "version": "1.1.0",
                "planned_date": "2024-03-15",
                "features": [
                    "Function calling support",
                    "JSON mode responses",
                    "Vision model integration",
                    "Improved rate limiting with burst"
                ]
            },
            {
                "version": "1.2.0",
                "planned_date": "2024-06-15",
                "features": [
                    "Fine-tuning API",
                    "Custom model deployment",
                    "A/B testing framework",
                    "Cost optimization recommendations"
                ]
            }
        ],
        "deprecation_policy": {
            "notice_period": "6 months",
            "grace_period": "3 months",
            "current_version_support_until": "2025-01-15"
        }
    }


# ============================================================================
# V1 API STATUS
# ============================================================================

@router.get(
    "/status",
    summary="API v1 Status",
    description="Get the current status of API v1 services",
    response_model=Dict[str, Any],
    include_in_schema=True
)
async def v1_api_status(request: Request):
    """
    Get the current operational status of all v1 API services.
    """
    from monitoring.health_check import HealthChecker
    
    health_checker = HealthChecker()
    
    # Check core services
    services = {
        "chat": await health_checker.check_service("chat"),
        "models": await health_checker.check_service("models"),
        "routing": await health_checker.check_service("routing"),
        "cache": await health_checker.check_service("cache"),
        "database": await health_checker.check_service("database")
    }
    
    # Determine overall status
    overall_status = "operational"
    degraded_services = []
    
    for service_name, service_status in services.items():
        if service_status.get("status") != "healthy":
            overall_status = "degraded"
            degraded_services.append(service_name)
    
    return {
        "version": "1.0.0",
        "status": overall_status,
        "timestamp": time.time(),
        "services": services,
        "degraded_services": degraded_services if degraded_services else None,
        "maintenance_mode": False,
        "scheduled_maintenance": None,
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


# ============================================================================
# V1 API CAPABILITIES
# ============================================================================

@router.get(
    "/capabilities",
    summary="API v1 Capabilities",
    description="Get the capabilities and features available in API v1",
    response_model=Dict[str, Any],
    include_in_schema=True
)
async def v1_capabilities():
    """
    Return the complete capabilities matrix for API v1.
    Used by clients to determine which features are available.
    """
    return {
        "version": "1.0.0",
        "capabilities": {
            "chat": {
                "supported": True,
                "streaming": settings.enable_streaming,
                "max_messages": 50,
                "max_tokens": 4096,
                "temperature_range": [0.0, 2.0],
                "top_p_range": [0.0, 1.0],
                "stop_sequences": True,
                "presence_penalty": True,
                "frequency_penalty": True,
                "logit_bias": False
            },
            "models": {
                "supported": True,
                "total_models": len(settings.model_config_dict.get("models", {})),
                "external_providers": [
                    "grok",
                    "openai", 
                    "anthropic",
                    "cohere"
                ],
                "local_formats": [
                    "transformers",
                    "gguf",
                    "safetensors"
                ],
                "quantization": [
                    "4bit",
                    "8bit",
                    "fp16",
                    "bf16"
                ]
            },
            "routing": {
                "supported": True,
                "strategies": [
                    "latency",
                    "cost", 
                    "quality",
                    "hybrid",
                    "round_robin"
                ],
                "weights_configurable": True,
                "fallback_enabled": settings.enable_fallbacks,
                "circuit_breaker": settings.enable_circuit_breaker
            },
            "caching": {
                "supported": settings.cache.enabled,
                "semantic": settings.cache.semantic_enabled,
                "similarity_threshold": settings.cache.similarity_threshold,
                "default_ttl": settings.cache.default_ttl,
                "max_size": settings.cache.max_size
            },
            "embeddings": {
                "supported": True,
                "models": ["all-MiniLM-L6-v2"],
                "dimensions": [384],
                "max_tokens": 512,
                "similarity_methods": ["cosine", "dot", "euclidean"]
            },
            "files": {
                "supported": True,
                "formats": ["pdf", "docx", "pptx", "txt", "md"],
                "max_size_mb": 25,
                "extraction_methods": ["text", "metadata"],
                "ocr": False
            },
            "batches": {
                "supported": True,
                "max_size": 100,
                "timeout_hours": 24,
                "retention_days": 7,
                "status_check": True,
                "webhooks": False
            },
            "monitoring": {
                "supported": True,
                "metrics": [
                    "requests_per_second",
                    "latency_p95",
                    "error_rate",
                    "token_usage",
                    "cache_hit_rate",
                    "cost_per_request"
                ],
                "real_time": True,
                "historical": True,
                "export_formats": ["json", "csv"]
            }
        },
        "limitations": {
            "max_prompt_length": 4096,
            "max_concurrent_requests": settings.server.max_concurrent_requests,
            "rate_limit_default": settings.rate_limit.default_requests,
            "file_size_limit_mb": 25,
            "batch_size_limit": 100,
            "embedding_dimensions": 384
        },
        "beta_features": [],
        "deprecated_features": []
    }


# ============================================================================
# V1 API SCHEMA
# ============================================================================

@router.get(
    "/schema",
    summary="API v1 Schema",
    description="Get the JSON schema for API v1 request/response models",
    response_model=Dict[str, Any],
    include_in_schema=False
)
async def v1_api_schema():
    """
    Return JSON schema definitions for all v1 API models.
    Useful for client code generation.
    """
    # This would generate JSON schema from Pydantic models
    # For now, return a placeholder
    return {
        "version": "1.0.0",
        "message": "Schema generation coming soon",
        "documentation_url": "/docs"
    }


# ============================================================================
# V1 API USAGE EXAMPLES
# ============================================================================

@router.get(
    "/examples",
    summary="API v1 Examples",
    description="Get example requests and responses for API v1 endpoints",
    response_model=Dict[str, Any],
    include_in_schema=True
)
async def v1_examples():
    """
    Return example API calls for common use cases.
    Helps developers get started quickly.
    """
    return {
        "version": "1.0.0",
        "examples": {
            "chat_completion": {
                "request": {
                    "method": "POST",
                    "url": "/api/v1/chat/completions",
                    "headers": {
                        "X-API-Key": "llm_your_api_key_here",
                        "Content-Type": "application/json"
                    },
                    "body": {
                        "messages": [
                            {"role": "user", "content": "What is the capital of France?"}
                        ],
                        "model": "gpt-3.5-turbo",
                        "stream": False,
                        "max_tokens": 100,
                        "temperature": 0.7
                    }
                },
                "response": {
                    "id": "chatcmpl-123abc",
                    "model": "gpt-3.5-turbo",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The capital of France is Paris."
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                        "total_tokens": 18
                    }
                }
            },
            "streaming_chat": {
                "request": {
                    "method": "POST",
                    "url": "/api/v1/chat/completions",
                    "headers": {
                        "X-API-Key": "llm_your_api_key_here",
                        "Content-Type": "application/json"
                    },
                    "body": {
                        "messages": [
                            {"role": "user", "content": "Write a haiku about AI"}
                        ],
                        "stream": True
                    }
                },
                "response": "data: {\"id\":\"chatcmpl-123\",\"choices\":[{\"delta\":{\"content\":\"Silent \"}}]}\n\ndata: {\"id\":\"chatcmpl-123\",\"choices\":[{\"delta\":{\"content\":\"circuits \"}}]}\n\ndata: [DONE]"
            },
            "list_models": {
                "request": {
                    "method": "GET",
                    "url": "/api/v1/models",
                    "headers": {
                        "X-API-Key": "llm_your_api_key_here"
                    }
                },
                "response": {
                    "object": "list",
                    "data": [
                        {
                            "id": "gpt-4",
                            "object": "model",
                            "owned_by": "openai",
                            "ready": True
                        }
                    ]
                }
            }
        }
    }


# ============================================================================
# V1 API PING
# ============================================================================

@router.get(
    "/ping",
    summary="API v1 Ping",
    description="Simple ping endpoint to test API v1 connectivity",
    response_model=Dict[str, Any],
    include_in_schema=True
)
async def v1_ping(request: Request):
    """
    Ultra-lightweight endpoint to test API connectivity.
    Returns immediately with minimal overhead.
    """
    return {
        "ping": "pong",
        "version": "1.0.0",
        "timestamp": time.time(),
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


# ============================================================================
# ERROR HANDLING FOR V1 ROUTER
# ============================================================================

@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    include_in_schema=False
)
async def v1_catch_all(request: Request, path: str):
    """
    Catch-all route for undefined v1 API endpoints.
    Returns a 404 error with helpful message.
    """
    logger.warning(
        "v1_endpoint_not_found",
        path=request.url.path,
        method=request.method,
        request_id=getattr(request.state, "request_id", None)
    )
    
    return HTTPException(
        status_code=404,
        detail={
            "error": "NotFound",
            "detail": f"API v1 endpoint '{request.method} {request.url.path}' not found",
            "version": "1.0.0",
            "available_endpoints": [
                "/chat",
                "/completions",
                "/models",
                "/routing",
                "/monitoring",
                "/admin",
                "/health",
                "/embeddings",
                "/files",
                "/batches"
            ],
            "documentation_url": "/docs",
            "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
            "timestamp": time.time()
        }
    )


# ============================================================================
# V1 API METRICS
# ============================================================================

@router.get(
    "/metrics",
    summary="API v1 Metrics",
    description="Get performance metrics for API v1 endpoints",
    response_model=Dict[str, Any],
    include_in_schema=True,
    dependencies=[Depends(verify_admin)]
)
async def v1_metrics(request: Request):
    """
    Get detailed performance metrics for all v1 API endpoints.
    Admin only - used for capacity planning and optimization.
    """
    from monitoring.metrics import MetricsCollector
    
    metrics_collector = MetricsCollector()
    
    # Get endpoint metrics
    endpoint_metrics = await metrics_collector.get_endpoint_metrics(version="v1")
    
    # Get aggregated metrics
    aggregated = await metrics_collector.get_aggregated_metrics(version="v1")
    
    return {
        "version": "1.0.0",
        "period": "24h",
        "timestamp": time.time(),
        "endpoints": endpoint_metrics,
        "aggregated": aggregated,
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["router"]