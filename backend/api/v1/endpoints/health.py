"""
Health Check Endpoints - Production Ready
Comprehensive health monitoring for all system components
Used by Kubernetes, load balancers, monitoring systems, and internal services
"""

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import time
import platform
import psutil
import uuid
from datetime import datetime, timedelta
import asyncio

from core.logging import get_logger
from core.exceptions import ServiceUnavailableError
from config import settings
from monitoring.health_check import HealthChecker
from database.session import check_db_connection
from cache.cache_manager import CacheManager
from models.model_manager import ModelManager

# Initialize router
router = APIRouter(prefix="/health", tags=["Health"])

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# HEALTH CHECK CONSTANTS
# ============================================================================

# Health status types
STATUS_HEALTHY = "healthy"
STATUS_DEGRADED = "degraded"
STATUS_UNHEALTHY = "unhealthy"
STATUS_MAINTENANCE = "maintenance"

# Service component names
COMPONENT_API = "api"
COMPONENT_DATABASE = "database"
COMPONENT_REDIS = "redis"
COMPONENT_CACHE = "cache"
COMPONENT_MODELS = "models"
COMPONENT_GPU = "gpu"
COMPONENT_DISK = "disk"
COMPONENT_MEMORY = "memory"

# ============================================================================
# HEALTH CHECKER INSTANCE
# ============================================================================

health_checker = HealthChecker()
cache_manager = CacheManager()
model_manager = ModelManager()

# ============================================================================
# BASIC HEALTH ENDPOINTS - NO AUTH REQUIRED
# ============================================================================

@router.get(
    "",
    summary="Health Check",
    description="""
    Comprehensive health check endpoint.
    
    Returns detailed health status of all system components:
    - API Gateway health
    - Database connectivity
    - Redis cache status
    - Model service status
    - GPU availability
    - Disk space
    - Memory usage
    
    Used by:
    - Kubernetes liveness/readiness probes
    - Load balancer health checks
    - Monitoring systems (Prometheus, Grafana)
    - Internal service discovery
    """,
    response_model=Dict[str, Any],
    status_code=200
)
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Comprehensive health check for all system components.
    
    This endpoint:
    1. Checks all critical services (database, cache, models)
    2. Returns detailed component status
    3. Includes system metrics (CPU, memory, disk)
    4. Adds request ID for tracing
    5. Returns 200 if all critical components are healthy
    6. Returns 503 if critical components are unhealthy
    """
    
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "health_check_started",
        request_id=request_id,
        client_ip=request.client.host if request.client else None
    )
    
    try:
        # ====================================================================
        # RUN ALL HEALTH CHECKS IN PARALLEL FOR PERFORMANCE
        # ====================================================================
        
        tasks = [
            check_api_health(),
            check_database_health(),
            check_redis_health(),
            check_cache_health(),
            check_models_health(),
            check_gpu_health(),
            check_system_resources()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        (
            api_status,
            db_status,
            redis_status,
            cache_status,
            models_status,
            gpu_status,
            system_status
        ) = results
        
        # Handle any exceptions in health checks
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "health_check_failed",
                    component=["api", "db", "redis", "cache", "models", "gpu", "system"][i],
                    error=str(result),
                    request_id=request_id
                )
                results[i] = {
                    "status": STATUS_UNHEALTHY,
                    "error": str(result),
                    "timestamp": time.time()
                }
        
        # ====================================================================
        # DETERMINE OVERALL HEALTH STATUS
        # ====================================================================
        
        # Critical components that must be healthy
        critical_components = [db_status, redis_status, models_status]
        critical_healthy = all(
            isinstance(c, dict) and c.get("status") == STATUS_HEALTHY
            for c in critical_components
        )
        
        # Overall status determination
        if not critical_healthy:
            overall_status = STATUS_UNHEALTHY
            status_code = 503  # Service Unavailable
        elif any(
            isinstance(c, dict) and c.get("status") == STATUS_DEGRADED
            for c in results
        ):
            overall_status = STATUS_DEGRADED
            status_code = 200  # Still usable but degraded
        else:
            overall_status = STATUS_HEALTHY
            status_code = 200
        
        # ====================================================================
        # BUILD RESPONSE
        # ====================================================================
        
        response = {
            "status": overall_status,
            "timestamp": time.time(),
            "request_id": request_id,
            "version": settings.version,
            "environment": settings.environment.value,
            "hostname": platform.node(),
            "uptime_seconds": get_uptime(),
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "components": {
                COMPONENT_API: api_status,
                COMPONENT_DATABASE: db_status,
                COMPONENT_REDIS: redis_status,
                COMPONENT_CACHE: cache_status,
                COMPONENT_MODELS: models_status,
                COMPONENT_GPU: gpu_status,
                COMPONENT_DISK: system_status.get("disk", {}),
                COMPONENT_MEMORY: system_status.get("memory", {})
            },
            "maintenance_mode": is_maintenance_mode(),
            "scheduled_maintenance": get_scheduled_maintenance()
        }
        
        # Add deprecation warnings if needed
        if settings.environment.is_production() and get_days_until_deprecation() < 180:
            response["deprecation_warning"] = {
                "message": "This API version will be deprecated",
                "deprecation_date": settings.deprecation_date,
                "days_remaining": get_days_until_deprecation(),
                "upgrade_url": "/docs"
            }
        
        # Log health check result
        logger.info(
            "health_check_completed",
            request_id=request_id,
            status=overall_status,
            response_time_ms=response["response_time_ms"],
            components_healthy=sum(
                1 for c in results 
                if isinstance(c, dict) and c.get("status") == STATUS_HEALTHY
            ),
            components_total=len(results)
        )
        
        # Return appropriate status code
        return JSONResponse(
            status_code=status_code,
            content=response
        )
        
    except Exception as e:
        logger.error(
            "health_check_error",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "status": STATUS_UNHEALTHY,
                "error": "Health check failed",
                "detail": str(e) if settings.debug else "Internal error",
                "timestamp": time.time(),
                "request_id": request_id
            }
        )


@router.get(
    "/live",
    summary="Liveness Probe",
    description="""
    Kubernetes liveness probe endpoint.
    
    Returns 200 if the service is alive, regardless of dependencies.
    Used by Kubernetes to know when to restart the container.
    
    This endpoint:
    - Never returns 5xx errors
    - Does NOT check dependencies
    - Sub-millisecond response time
    - No authentication required
    """,
    response_model=Dict[str, str],
    status_code=200
)
async def liveness_probe(request: Request) -> Dict[str, str]:
    """
    Lightweight liveness probe for Kubernetes.
    
    This endpoint should:
    - Be extremely fast (<1ms)
    - Never fail (unless process is dead)
    - Have no external dependencies
    - Not be rate limited
    """
    return {
        "status": "alive",
        "timestamp": str(time.time()),
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


@router.get(
    "/ready",
    summary="Readiness Probe",
    description="""
    Kubernetes readiness probe endpoint.
    
    Returns 200 if the service is ready to accept traffic.
    Used by Kubernetes to know when to send traffic to the pod.
    
    This endpoint:
    - Checks critical dependencies
    - Returns 503 if not ready
    - Has timeout protection
    - No authentication required
    """,
    response_model=Dict[str, Any],
    status_code=200
)
async def readiness_probe(request: Request) -> Dict[str, Any]:
    """
    Readiness probe for Kubernetes.
    
    Checks if the service is ready to accept traffic:
    1. Database connection
    2. Redis connection
    3. Cache initialized
    4. Models loaded (minimal set)
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # Check database with timeout
    try:
        db_ready = await asyncio.wait_for(
            check_db_connection(),
            timeout=2.0
        )
    except (asyncio.TimeoutError, Exception):
        db_ready = False
    
    # Check Redis with timeout
    try:
        redis_ready = await asyncio.wait_for(
            cache_manager.ping(),
            timeout=1.0
        )
    except (asyncio.TimeoutError, Exception):
        redis_ready = False
    
    # Check if at least one model is loaded
    try:
        loaded_models = await asyncio.wait_for(
            model_manager.get_loaded_models_count(),
            timeout=2.0
        )
        models_ready = loaded_models > 0
    except (asyncio.TimeoutError, Exception):
        models_ready = False
    
    # Determine readiness
    is_ready = db_ready and redis_ready and models_ready
    
    if not is_ready:
        logger.warning(
            "readiness_probe_failed",
            request_id=request_id,
            db_ready=db_ready,
            redis_ready=redis_ready,
            models_ready=models_ready
        )
        
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "timestamp": time.time(),
                "request_id": request_id,
                "checks": {
                    "database": "ready" if db_ready else "not_ready",
                    "redis": "ready" if redis_ready else "not_ready",
                    "models": "ready" if models_ready else "not_ready"
                }
            }
        )
    
    return {
        "status": "ready",
        "timestamp": time.time(),
        "request_id": request_id,
        "checks": {
            "database": "ready",
            "redis": "ready", 
            "models": f"{loaded_models} models loaded"
        }
    }


@router.get(
    "/startup",
    summary="Startup Probe",
    description="""
    Kubernetes startup probe endpoint.
    
    Returns 200 when the application has completed initialization.
    Used by Kubernetes for slow-starting applications.
    
    This endpoint:
    - Checks if all startup tasks completed
    - Returns 503 during initialization
    - Has longer timeout than readiness probe
    """,
    response_model=Dict[str, Any],
    status_code=200
)
async def startup_probe(request: Request) -> Dict[str, Any]:
    """
    Startup probe for slow-initializing applications.
    
    Checks if the application has completed its startup sequence:
    1. Database migrations complete
    2. Cache warmed up
    3. Models pre-warmed (if configured)
    4. Background tasks started
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # Get startup status from global state
    from main import startup_complete, startup_errors
    
    if not startup_complete:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "timestamp": time.time(),
                "request_id": request_id,
                "errors": startup_errors if startup_errors else None,
                "progress": get_startup_progress()
            }
        )
    
    return {
        "status": "started",
        "timestamp": time.time(),
        "request_id": request_id,
        "startup_time_seconds": get_startup_time(),
        "components_loaded": get_loaded_components()
    }


# ============================================================================
# DETAILED COMPONENT HEALTH CHECKS
# ============================================================================

@router.get(
    "/database",
    summary="Database Health",
    description="Detailed PostgreSQL database health check",
    response_model=Dict[str, Any]
)
async def database_health(request: Request) -> Dict[str, Any]:
    """Check PostgreSQL database health with detailed metrics."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    start_time = time.time()
    
    try:
        result = await check_database_health()
        
        return {
            "status": result["status"],
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": time.time(),
            "request_id": request_id,
            **result
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        
        return JSONResponse(
            status_code=503,
            content={
                "status": STATUS_UNHEALTHY,
                "error": str(e),
                "timestamp": time.time(),
                "request_id": request_id
            }
        )


@router.get(
    "/cache",
    summary="Cache Health",
    description="Detailed Redis cache health check",
    response_model=Dict[str, Any]
)
async def cache_health(request: Request) -> Dict[str, Any]:
    """Check Redis cache health with detailed metrics."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    start_time = time.time()
    
    try:
        result = await check_cache_health()
        
        return {
            "status": result["status"],
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": time.time(),
            "request_id": request_id,
            **result
        }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        
        return JSONResponse(
            status_code=503,
            content={
                "status": STATUS_UNHEALTHY,
                "error": str(e),
                "timestamp": time.time(),
                "request_id": request_id
            }
        )


@router.get(
    "/models",
    summary="Models Health",
    description="Detailed model service health check",
    response_model=Dict[str, Any]
)
async def models_health(request: Request) -> Dict[str, Any]:
    """Check model service health with detailed metrics."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    start_time = time.time()
    
    try:
        result = await check_models_health()
        
        return {
            "status": result["status"],
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": time.time(),
            "request_id": request_id,
            **result
        }
    except Exception as e:
        logger.error(f"Models health check failed: {e}")
        
        return JSONResponse(
            status_code=503,
            content={
                "status": STATUS_UNHEALTHY,
                "error": str(e),
                "timestamp": time.time(),
                "request_id": request_id
            }
        )


@router.get(
    "/gpu",
    summary="GPU Health",
    description="Detailed GPU health check",
    response_model=Dict[str, Any]
)
async def gpu_health(request: Request) -> Dict[str, Any]:
    """Check GPU health with detailed metrics (if available)."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    result = await check_gpu_health()
    
    return {
        "timestamp": time.time(),
        "request_id": request_id,
        **result
    }


@router.get(
    "/system",
    summary="System Health",
    description="Detailed system resource health check",
    response_model=Dict[str, Any]
)
async def system_health(request: Request) -> Dict[str, Any]:
    """Check system resource health (CPU, memory, disk)."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    result = await check_system_resources()
    
    return {
        "timestamp": time.time(),
        "request_id": request_id,
        **result
    }


# ============================================================================
# HEALTH CHECK IMPLEMENTATIONS
# ============================================================================

async def check_api_health() -> Dict[str, Any]:
    """Check API gateway health."""
    return {
        "status": STATUS_HEALTHY,
        "version": settings.version,
        "environment": settings.environment.value,
        "debug": settings.debug,
        "timestamp": time.time()
    }


async def check_database_health() -> Dict[str, Any]:
    """Check PostgreSQL database health with detailed metrics."""
    start_time = time.time()
    
    try:
        # Test connection
        conn = await check_db_connection()
        
        # Get connection pool stats
        from database.session import get_pool_stats
        pool_stats = await get_pool_stats()
        
        # Get database version and size
        from database.session import get_db_info
        db_info = await get_db_info()
        
        response_time = (time.time() - start_time) * 1000
        
        status = STATUS_HEALTHY
        if response_time > 100:
            status = STATUS_DEGRADED
        if not conn:
            status = STATUS_UNHEALTHY
        
        return {
            "status": status,
            "connected": conn,
            "response_time_ms": round(response_time, 2),
            "pool": pool_stats,
            "info": db_info,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": STATUS_UNHEALTHY,
            "connected": False,
            "error": str(e),
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": time.time()
        }


async def check_redis_health() -> Dict[str, Any]:
    """Check Redis cache health with detailed metrics."""
    start_time = time.time()
    
    try:
        # Ping Redis
        ping_result = await cache_manager.ping()
        
        # Get Redis info
        info = await cache_manager.get_info()
        
        # Get cache stats
        stats = await cache_manager.get_stats()
        
        response_time = (time.time() - start_time) * 1000
        
        status = STATUS_HEALTHY
        if response_time > 50:
            status = STATUS_DEGRADED
        if not ping_result:
            status = STATUS_UNHEALTHY
        
        return {
            "status": status,
            "connected": ping_result,
            "response_time_ms": round(response_time, 2),
            "version": info.get("redis_version", "unknown"),
            "used_memory_mb": info.get("used_memory_rss", 0) / (1024 * 1024),
            "total_memory_mb": info.get("maxmemory", 0) / (1024 * 1024),
            "connected_clients": info.get("connected_clients", 0),
            "cache_hits": stats.get("hits", 0),
            "cache_misses": stats.get("misses", 0),
            "hit_rate": stats.get("hit_rate", 0),
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": STATUS_UNHEALTHY,
            "connected": False,
            "error": str(e),
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": time.time()
        }


async def check_cache_health() -> Dict[str, Any]:
    """Check application cache health."""
    start_time = time.time()
    
    try:
        # Test cache operations
        test_key = f"health_check_{uuid.uuid4()}"
        test_value = {"test": "data"}
        
        await cache_manager.set(test_key, test_value, ttl=10)
        retrieved = await cache_manager.get(test_key)
        await cache_manager.delete(test_key)
        
        response_time = (time.time() - start_time) * 1000
        working = retrieved == test_value
        
        status = STATUS_HEALTHY if working else STATUS_UNHEALTHY
        
        return {
            "status": status,
            "working": working,
            "response_time_ms": round(response_time, 2),
            "semantic_enabled": settings.cache.semantic_enabled,
            "similarity_threshold": settings.cache.similarity_threshold,
            "default_ttl": settings.cache.default_ttl,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": STATUS_UNHEALTHY,
            "working": False,
            "error": str(e),
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": time.time()
        }


async def check_models_health() -> Dict[str, Any]:
    """Check model service health with detailed metrics."""
    start_time = time.time()
    
    try:
        # Get model statistics
        loaded_models = await model_manager.get_loaded_models()
        loaded_count = len(loaded_models)
        total_models = len(settings.model_config_dict.get("models", {}))
        
        # Check if any models are loaded
        status = STATUS_HEALTHY
        if loaded_count == 0:
            status = STATUS_DEGRADED
        
        # Check if any models are in error state
        error_models = [m for m in loaded_models if m.get("status") == "error"]
        if error_models:
            status = STATUS_DEGRADED
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": status,
            "response_time_ms": round(response_time, 2),
            "loaded_count": loaded_count,
            "total_configured": total_models,
            "error_count": len(error_models),
            "error_models": [m["id"] for m in error_models] if error_models else [],
            "models": [
                {
                    "id": m["id"],
                    "name": m.get("name", m["id"]),
                    "type": m.get("type", "unknown"),
                    "status": m.get("status", "unknown"),
                    "loaded_at": m.get("loaded_at"),
                    "memory_mb": m.get("memory_mb", 0)
                }
                for m in loaded_models[:10]  # Limit to 10 for response size
            ],
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": STATUS_UNHEALTHY,
            "error": str(e),
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": time.time()
        }


async def check_gpu_health() -> Dict[str, Any]:
    """Check GPU health if available."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                "status": STATUS_HEALTHY,
                "available": False,
                "message": "No GPU detected, running on CPU",
                "timestamp": time.time()
            }
        
        # Get GPU information
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        
        for i in range(gpu_count):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            gpu_cached = torch.cuda.memory_reserved(i) / (1024**3)
            
            gpu_info.append({
                "index": i,
                "name": gpu_props.name,
                "memory_total_gb": round(gpu_memory, 2),
                "memory_used_gb": round(gpu_allocated, 2),
                "memory_cached_gb": round(gpu_cached, 2),
                "memory_free_gb": round(gpu_memory - gpu_allocated, 2),
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "cuda_version": torch.version.cuda
            })
        
        return {
            "status": STATUS_HEALTHY,
            "available": True,
            "count": gpu_count,
            "devices": gpu_info,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "timestamp": time.time()
        }
        
    except ImportError:
        return {
            "status": STATUS_HEALTHY,
            "available": False,
            "message": "PyTorch not installed, GPU monitoring unavailable",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": STATUS_DEGRADED,
            "available": False,
            "error": str(e),
            "timestamp": time.time()
        }


async def check_system_resources() -> Dict[str, Any]:
    """Check system resource health (CPU, memory, disk)."""
    
    # CPU information
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    cpu_status = STATUS_HEALTHY
    if cpu_percent > 80:
        cpu_status = STATUS_DEGRADED
    if cpu_percent > 95:
        cpu_status = STATUS_UNHEALTHY
    
    # Memory information
    memory = psutil.virtual_memory()
    
    memory_status = STATUS_HEALTHY
    if memory.percent > 80:
        memory_status = STATUS_DEGRADED
    if memory.percent > 90:
        memory_status = STATUS_UNHEALTHY
    
    # Disk information
    disk = psutil.disk_usage("/")
    
    disk_status = STATUS_HEALTHY
    if disk.percent > 80:
        disk_status = STATUS_DEGRADED
    if disk.percent > 90:
        disk_status = STATUS_UNHEALTHY
    
    return {
        "cpu": {
            "status": cpu_status,
            "percent": cpu_percent,
            "count": cpu_count,
            "frequency_mhz": cpu_freq.current if cpu_freq else None,
            "load_avg": [round(l, 2) for l in psutil.getloadavg()] if hasattr(psutil, "getloadavg") else None
        },
        "memory": {
            "status": memory_status,
            "percent": memory.percent,
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "free_gb": round(memory.free / (1024**3), 2)
        },
        "disk": {
            "status": disk_status,
            "percent": disk.percent,
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2)
        },
        "timestamp": time.time()
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_uptime() -> float:
    """Get system uptime in seconds."""
    try:
        return time.time() - psutil.boot_time()
    except:
        return 0


def get_startup_time() -> float:
    """Get application startup time in seconds."""
    try:
        from main import startup_time
        return startup_time
    except:
        return 0


def get_startup_progress() -> Dict[str, Any]:
    """Get application startup progress."""
    try:
        from main import startup_progress
        return startup_progress
    except:
        return {}


def get_loaded_components() -> List[str]:
    """Get list of loaded components."""
    try:
        from main import loaded_components
        return loaded_components
    except:
        return []


def is_maintenance_mode() -> bool:
    """Check if maintenance mode is enabled."""
    # TODO: Implement maintenance mode toggle
    return False


def get_scheduled_maintenance() -> Optional[Dict[str, Any]]:
    """Get scheduled maintenance information."""
    # TODO: Implement scheduled maintenance
    return None


def get_days_until_deprecation() -> int:
    """Get days until API deprecation."""
    if hasattr(settings, "deprecation_date") and settings.deprecation_date:
        days = (settings.deprecation_date - datetime.now().date()).days
        return max(0, days)
    return 365  # Default 1 year


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "router",
    "STATUS_HEALTHY",
    "STATUS_DEGRADED", 
    "STATUS_UNHEALTHY",
    "STATUS_MAINTENANCE"
]