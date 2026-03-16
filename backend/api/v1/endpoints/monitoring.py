"""
Monitoring & Observability Endpoints - Production Ready
Real-time metrics, health dashboards, logs, traces, and alerts
Complete system observability for LLM Gateway operations
"""

from fastapi import APIRouter, Request, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List, Optional, Union
import time
import uuid
import asyncio
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from core.logging import get_logger
from core.security import verify_api_key, verify_admin, get_current_user
from core.rate_limiter import rate_limit
from config import settings
from monitoring.metrics import MetricsCollector
from monitoring.health_check import HealthChecker
from monitoring.alerting import AlertManager
from monitoring.log_aggregator import LogAggregator
from database.repositories.metrics_repository import MetricsRepository
from database.repositories.request_log_repository import RequestLogRepository

# Initialize router
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# Initialize logger
logger = get_logger(__name__)

# Initialize services
metrics_collector = MetricsCollector()
health_checker = HealthChecker()
alert_manager = AlertManager()
log_aggregator = LogAggregator()
metrics_repository = MetricsRepository()
request_log_repository = RequestLogRepository()

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class MetricPoint(BaseModel):
    """Single metric data point"""
    timestamp: datetime = Field(..., description="Metric timestamp")
    value: float = Field(..., description="Metric value")
    tags: Optional[Dict[str, str]] = Field(None, description="Metric tags")


class TimeSeriesResponse(BaseModel):
    """Time series metrics response"""
    metric: str = Field(..., description="Metric name")
    unit: str = Field(..., description="Metric unit")
    data: List[MetricPoint] = Field(..., description="Metric data points")
    summary: Dict[str, float] = Field(..., description="Summary statistics")


class SystemMetricsResponse(BaseModel):
    """System metrics response"""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    cpu: Dict[str, Any] = Field(..., description="CPU metrics")
    memory: Dict[str, Any] = Field(..., description="Memory metrics")
    disk: Dict[str, Any] = Field(..., description="Disk metrics")
    network: Dict[str, Any] = Field(..., description="Network metrics")
    gpu: Optional[Dict[str, Any]] = Field(None, description="GPU metrics")


class RequestMetricsResponse(BaseModel):
    """Request metrics response"""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    total_requests: int = Field(..., description="Total requests")
    requests_per_second: float = Field(..., description="Requests per second")
    average_latency_ms: float = Field(..., description="Average latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    p99_latency_ms: float = Field(..., description="99th percentile latency")
    error_rate: float = Field(..., description="Error rate percentage")
    status_codes: Dict[str, int] = Field(..., description="Status code distribution")
    top_endpoints: List[Dict[str, Any]] = Field(..., description="Top endpoints")


class ModelMetricsResponse(BaseModel):
    """Model metrics response"""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    model_id: str = Field(..., description="Model identifier")
    total_requests: int = Field(..., description="Total requests")
    success_count: int = Field(..., description="Successful requests")
    error_count: int = Field(..., description="Failed requests")
    avg_latency_ms: float = Field(..., description="Average latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    p99_latency_ms: float = Field(..., description="99th percentile latency")
    total_tokens: int = Field(..., description="Total tokens generated")
    avg_tokens_per_request: float = Field(..., description="Average tokens per request")
    tokens_per_second: float = Field(..., description="Tokens per second")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage")
    gpu_usage_percent: Optional[float] = Field(None, description="GPU utilization")
    cost_total: float = Field(0, description="Total cost in USD")
    cost_per_request: float = Field(0, description="Average cost per request")


class CacheMetricsResponse(BaseModel):
    """Cache metrics response"""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    hits: int = Field(..., description="Cache hits")
    misses: int = Field(..., description="Cache misses")
    hit_rate: float = Field(..., description="Cache hit rate")
    size_bytes: int = Field(..., description="Cache size in bytes")
    entries: int = Field(..., description="Number of cache entries")
    evictions: int = Field(..., description="Cache evictions")
    avg_lookup_time_ms: float = Field(..., description="Average lookup time")
    semantic_hits: Optional[int] = Field(None, description="Semantic cache hits")
    semantic_misses: Optional[int] = Field(None, description="Semantic cache misses")
    semantic_hit_rate: Optional[float] = Field(None, description="Semantic cache hit rate")


class LogEntry(BaseModel):
    """Log entry schema"""
    timestamp: datetime = Field(..., description="Log timestamp")
    level: str = Field(..., description="Log level")
    message: str = Field(..., description="Log message")
    request_id: Optional[str] = Field(None, description="Request ID")
    service: str = Field(..., description="Service name")
    component: Optional[str] = Field(None, description="Component name")
    user_id: Optional[str] = Field(None, description="User ID")
    model_id: Optional[str] = Field(None, description="Model ID")
    endpoint: Optional[str] = Field(None, description="API endpoint")
    status_code: Optional[int] = Field(None, description="HTTP status code")
    latency_ms: Optional[float] = Field(None, description="Request latency")
    error: Optional[str] = Field(None, description="Error message")
    traceback: Optional[str] = Field(None, description="Error traceback")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class LogsResponse(BaseModel):
    """Logs response schema"""
    total: int = Field(..., description="Total logs matching criteria")
    logs: List[LogEntry] = Field(..., description="Log entries")
    next_cursor: Optional[str] = Field(None, description="Cursor for next page")
    query: Dict[str, Any] = Field(..., description="Query parameters")


class Alert(BaseModel):
    """Alert schema"""
    id: str = Field(..., description="Alert ID")
    name: str = Field(..., description="Alert name")
    severity: str = Field(..., description="Severity (info, warning, critical)")
    status: str = Field(..., description="Status (firing, resolved, acknowledged)")
    message: str = Field(..., description="Alert message")
    description: Optional[str] = Field(None, description="Detailed description")
    affected_service: Optional[str] = Field(None, description="Affected service")
    affected_model: Optional[str] = Field(None, description="Affected model")
    value: Optional[float] = Field(None, description="Metric value")
    threshold: Optional[float] = Field(None, description="Alert threshold")
    started_at: datetime = Field(..., description="When alert started")
    resolved_at: Optional[datetime] = Field(None, description="When alert resolved")
    acknowledged_by: Optional[str] = Field(None, description="Who acknowledged")
    acknowledged_at: Optional[datetime] = Field(None, description="When acknowledged")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AlertsResponse(BaseModel):
    """Alerts response schema"""
    total: int = Field(..., description="Total alerts")
    firing: int = Field(..., description="Currently firing alerts")
    acknowledged: int = Field(..., description="Acknowledged alerts")
    alerts: List[Alert] = Field(..., description="Alert list")


class DashboardResponse(BaseModel):
    """Dashboard metrics response"""
    timestamp: datetime = Field(..., description="Dashboard timestamp")
    system: SystemMetricsResponse = Field(..., description="System metrics")
    requests: RequestMetricsResponse = Field(..., description="Request metrics")
    models: List[ModelMetricsResponse] = Field(..., description="Model metrics")
    cache: CacheMetricsResponse = Field(..., description="Cache metrics")
    alerts: Dict[str, int] = Field(..., description="Alert counts by severity")


class HealthStatus(BaseModel):
    """Health status response"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Status timestamp")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment")


# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

@router.get(
    "/metrics",
    summary="Get All Metrics",
    description="""
    Get comprehensive system metrics.
    
    Returns:
    - System metrics (CPU, memory, disk, network, GPU)
    - Request metrics (RPS, latency, errors)
    - Model metrics (performance, tokens, cost)
    - Cache metrics (hit rate, size, evictions)
    
    Supports time range filtering and aggregation.
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def get_all_metrics(
    request: Request,
    period: str = Query("1h", description="Time period (1h, 6h, 24h, 7d, 30d)"),
    resolution: str = Query("1m", description="Resolution (1m, 5m, 1h, 1d)"),
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Get all system metrics with time range filtering.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "metrics_requested",
        request_id=request_id,
        period=period,
        resolution=resolution
    )
    
    try:
        # Calculate time range
        end_time = datetime.now()
        
        if period == "1h":
            start_time_dt = end_time - timedelta(hours=1)
        elif period == "6h":
            start_time_dt = end_time - timedelta(hours=6)
        elif period == "24h":
            start_time_dt = end_time - timedelta(days=1)
        elif period == "7d":
            start_time_dt = end_time - timedelta(days=7)
        elif period == "30d":
            start_time_dt = end_time - timedelta(days=30)
        else:
            start_time_dt = end_time - timedelta(hours=1)
        
        # Get metrics from collector
        metrics = await metrics_collector.get_aggregated_metrics(
            start_time=start_time_dt,
            end_time=end_time,
            resolution=resolution
        )
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "period": period,
            "resolution": resolution,
            "request_id": request_id,
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            **metrics
        }
        
        logger.info(
            "metrics_completed",
            request_id=request_id,
            response_time_ms=response["response_time_ms"]
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "metrics_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve metrics",
                "request_id": request_id
            }
        )


@router.get(
    "/metrics/system",
    summary="Get System Metrics",
    description="""
    Get real-time system resource metrics.
    
    Returns:
    - CPU: usage, load average, frequency
    - Memory: total, used, free, swap
    - Disk: total, used, free per mount
    - Network: bytes in/out, packets, errors
    - GPU: utilization, memory, temperature (if available)
    
    Updates in real-time.
    """,
    response_model=SystemMetricsResponse,
    dependencies=[Depends(rate_limit(limit=120, period=60))]
)
async def get_system_metrics(
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> SystemMetricsResponse:
    """
    Get real-time system resource metrics.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        # Get system metrics from collector
        metrics = await metrics_collector.get_system_metrics()
        
        response = SystemMetricsResponse(
            timestamp=datetime.now(),
            **metrics
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "system_metrics_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve system metrics",
                "request_id": request_id
            }
        )


@router.get(
    "/metrics/requests",
    summary="Get Request Metrics",
    description="""
    Get API request metrics.
    
    Returns:
    - Request volume over time
    - Latency percentiles (p50, p95, p99)
    - Error rates and status code distribution
    - Top endpoints by request count and latency
    - User/API key usage statistics
    
    Supports time range filtering.
    """,
    response_model=RequestMetricsResponse,
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def get_request_metrics(
    request: Request,
    period: str = Query("1h", description="Time period (1h, 6h, 24h, 7d)"),
    api_key: str = Depends(verify_api_key)
) -> RequestMetricsResponse:
    """
    Get API request metrics with time range filtering.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        # Calculate time range
        end_time = datetime.now()
        
        if period == "1h":
            start_time = end_time - timedelta(hours=1)
        elif period == "6h":
            start_time = end_time - timedelta(hours=6)
        elif period == "24h":
            start_time = end_time - timedelta(days=1)
        elif period == "7d":
            start_time = end_time - timedelta(days=7)
        else:
            start_time = end_time - timedelta(hours=1)
        
        # Get metrics from repository
        metrics = await metrics_repository.get_request_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        # Get top endpoints
        top_endpoints = await request_log_repository.get_top_endpoints(
            start_time=start_time,
            end_time=end_time,
            limit=10
        )
        
        response = RequestMetricsResponse(
            timestamp=datetime.now(),
            total_requests=metrics.get("total_requests", 0),
            requests_per_second=metrics.get("requests_per_second", 0),
            average_latency_ms=metrics.get("avg_latency_ms", 0),
            p95_latency_ms=metrics.get("p95_latency_ms", 0),
            p99_latency_ms=metrics.get("p99_latency_ms", 0),
            error_rate=metrics.get("error_rate", 0),
            status_codes=metrics.get("status_codes", {}),
            top_endpoints=top_endpoints
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "request_metrics_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve request metrics",
                "request_id": request_id
            }
        )


@router.get(
    "/metrics/cache",
    summary="Get Cache Metrics",
    description="""
    Get cache performance metrics.
    
    Returns:
    - Cache hit/miss rates
    - Cache size and entry count
    - Eviction rate
    - Lookup latency
    - Semantic cache performance (if enabled)
    
    Supports time range filtering.
    """,
    response_model=CacheMetricsResponse,
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def get_cache_metrics(
    request: Request,
    period: str = Query("1h", description="Time period (1h, 6h, 24h, 7d)"),
    api_key: str = Depends(verify_api_key)
) -> CacheMetricsResponse:
    """
    Get cache performance metrics.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        # Calculate time range
        end_time = datetime.now()
        
        if period == "1h":
            start_time = end_time - timedelta(hours=1)
        elif period == "6h":
            start_time = end_time - timedelta(hours=6)
        elif period == "24h":
            start_time = end_time - timedelta(days=1)
        elif period == "7d":
            start_time = end_time - timedelta(days=7)
        else:
            start_time = end_time - timedelta(hours=1)
        
        # Get metrics from collector
        metrics = await metrics_collector.get_cache_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        response = CacheMetricsResponse(
            timestamp=datetime.now(),
            hits=metrics.get("hits", 0),
            misses=metrics.get("misses", 0),
            hit_rate=metrics.get("hit_rate", 0),
            size_bytes=metrics.get("size_bytes", 0),
            entries=metrics.get("entries", 0),
            evictions=metrics.get("evictions", 0),
            avg_lookup_time_ms=metrics.get("avg_lookup_time_ms", 0),
            semantic_hits=metrics.get("semantic_hits"),
            semantic_misses=metrics.get("semantic_misses"),
            semantic_hit_rate=metrics.get("semantic_hit_rate")
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "cache_metrics_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve cache metrics",
                "request_id": request_id
            }
        )


@router.get(
    "/metrics/models",
    summary="Get All Model Metrics",
    description="""
    Get performance metrics for all models.
    
    Returns aggregated metrics for all models:
    - Request counts and latency
    - Token usage and throughput
    - Error rates
    - Memory and GPU usage
    - Cost analysis
    """,
    response_model=List[ModelMetricsResponse],
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def get_all_model_metrics(
    request: Request,
    period: str = Query("24h", description="Time period (1h, 6h, 24h, 7d)"),
    api_key: str = Depends(verify_api_key)
) -> List[ModelMetricsResponse]:
    """
    Get performance metrics for all models.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        # Calculate time range
        end_time = datetime.now()
        
        if period == "1h":
            start_time = end_time - timedelta(hours=1)
        elif period == "6h":
            start_time = end_time - timedelta(hours=6)
        elif period == "24h":
            start_time = end_time - timedelta(days=1)
        elif period == "7d":
            start_time = end_time - timedelta(days=7)
        else:
            start_time = end_time - timedelta(hours=1)
        
        # Get metrics from collector
        metrics_list = await metrics_collector.get_all_model_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        responses = []
        for metrics in metrics_list:
            responses.append(ModelMetricsResponse(
                timestamp=datetime.now(),
                **metrics
            ))
        
        return responses
        
    except Exception as e:
        logger.error(
            "all_model_metrics_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve model metrics",
                "request_id": request_id
            }
        )


@router.get(
    "/metrics/models/{model_id}",
    summary="Get Model Metrics",
    description="""
    Get performance metrics for a specific model.
    
    Returns:
    - Request volume over time
    - Latency percentiles
    - Token usage and throughput
    - Error rates
    - Memory and GPU usage
    - Cost analysis
    
    Supports time range filtering.
    """,
    response_model=ModelMetricsResponse,
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def get_model_metrics(
    request: Request,
    model_id: str,
    period: str = Query("24h", description="Time period (1h, 6h, 24h, 7d)"),
    api_key: str = Depends(verify_api_key)
) -> ModelMetricsResponse:
    """
    Get performance metrics for a specific model.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "model_metrics_requested",
        request_id=request_id,
        model_id=model_id,
        period=period
    )
    
    try:
        # Calculate time range
        end_time = datetime.now()
        
        if period == "1h":
            start_time = end_time - timedelta(hours=1)
        elif period == "6h":
            start_time = end_time - timedelta(hours=6)
        elif period == "24h":
            start_time = end_time - timedelta(days=1)
        elif period == "7d":
            start_time = end_time - timedelta(days=7)
        else:
            start_time = end_time - timedelta(hours=1)
        
        # Get metrics from collector
        metrics = await metrics_collector.get_model_metrics(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time
        )
        
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ModelNotFound",
                    "message": f"No metrics found for model '{model_id}'",
                    "request_id": request_id
                }
            )
        
        response = ModelMetricsResponse(
            timestamp=datetime.now(),
            model_id=model_id,
            **metrics
        )
        
        logger.info(
            "model_metrics_completed",
            request_id=request_id,
            model_id=model_id,
            request_count=response.total_requests
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "model_metrics_failed",
            request_id=request_id,
            model_id=model_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to retrieve metrics for model '{model_id}'",
                "request_id": request_id
            }
        )


@router.get(
    "/metrics/timeseries/{metric_name}",
    summary="Get Time Series Metrics",
    description="""
    Get time series data for a specific metric.
    
    Returns raw time series data for custom analysis and visualization.
    
    Supported metrics:
    - requests.total, requests.success, requests.error
    - latency.avg, latency.p95, latency.p99
    - tokens.total, tokens.prompt, tokens.completion
    - cache.hits, cache.misses, cache.hit_rate
    - memory.used, memory.percent
    - gpu.utilization, gpu.memory
    
    Supports aggregation and downsampling.
    """,
    response_model=TimeSeriesResponse,
    dependencies=[Depends(rate_limit(limit=120, period=60))]
)
async def get_timeseries_metrics(
    request: Request,
    metric_name: str,
    period: str = Query("24h", description="Time period (1h, 6h, 24h, 7d, 30d)"),
    resolution: str = Query("5m", description="Resolution (1m, 5m, 1h, 1d)"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint"),
    api_key: str = Depends(verify_api_key)
) -> TimeSeriesResponse:
    """
    Get time series data for a specific metric.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        # Calculate time range
        end_time = datetime.now()
        
        if period == "1h":
            start_time = end_time - timedelta(hours=1)
        elif period == "6h":
            start_time = end_time - timedelta(hours=6)
        elif period == "24h":
            start_time = end_time - timedelta(days=1)
        elif period == "7d":
            start_time = end_time - timedelta(days=7)
        elif period == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(hours=1)
        
        # Parse resolution
        resolution_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "6h": 21600,
            "1d": 86400
        }
        
        interval = resolution_map.get(resolution, 300)
        
        # Get time series data
        data = await metrics_collector.get_timeseries(
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            model_id=model_id,
            endpoint=endpoint
        )
        
        # Calculate summary statistics
        values = [point["value"] for point in data]
        
        summary = {
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "avg": sum(values) / len(values) if values else 0,
            "current": values[-1] if values else 0
        }
        
        response = TimeSeriesResponse(
            metric=metric_name,
            unit=data.get("unit", "count"),
            data=[
                MetricPoint(
                    timestamp=point["timestamp"],
                    value=point["value"],
                    tags=point.get("tags")
                )
                for point in data.get("points", [])
            ],
            summary=summary
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "timeseries_metrics_failed",
            request_id=request_id,
            metric_name=metric_name,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to retrieve time series for metric '{metric_name}'",
                "request_id": request_id
            }
        )


# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

@router.get(
    "/dashboard",
    summary="Get Dashboard Metrics",
    description="""
    Get comprehensive dashboard metrics in a single response.
    
    Optimized for frontend dashboards:
    - All system metrics
    - Request performance
    - Top models
    - Cache status
    - Active alerts
    
    Returns everything needed for a real-time monitoring dashboard.
    """,
    response_model=DashboardResponse,
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def get_dashboard(
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> DashboardResponse:
    """
    Get comprehensive dashboard metrics.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "dashboard_requested",
        request_id=request_id
    )
    
    try:
        # Run all dashboard queries in parallel
        tasks = [
            metrics_collector.get_system_metrics(),
            metrics_collector.get_request_metrics(start_time=datetime.now() - timedelta(hours=1), end_time=datetime.now()),
            metrics_collector.get_all_model_metrics(start_time=datetime.now() - timedelta(hours=24), end_time=datetime.now()),
            metrics_collector.get_cache_metrics(start_time=datetime.now() - timedelta(hours=24), end_time=datetime.now()),
            alert_manager.get_alert_counts()
        ]
        
        system_metrics, request_metrics, model_metrics_list, cache_metrics, alert_counts = await asyncio.gather(*tasks)
        
        # Get top 5 models by request count
        model_metrics_list.sort(key=lambda x: x.get("total_requests", 0), reverse=True)
        top_models = model_metrics_list[:5]
        
        response = DashboardResponse(
            timestamp=datetime.now(),
            system=SystemMetricsResponse(
                timestamp=datetime.now(),
                **system_metrics
            ),
            requests=RequestMetricsResponse(
                timestamp=datetime.now(),
                **request_metrics,
                top_endpoints=await request_log_repository.get_top_endpoints(
                    start_time=datetime.now() - timedelta(hours=1),
                    end_time=datetime.now(),
                    limit=5
                )
            ),
            models=[
                ModelMetricsResponse(
                    timestamp=datetime.now(),
                    **model_metrics
                )
                for model_metrics in top_models
            ],
            cache=CacheMetricsResponse(
                timestamp=datetime.now(),
                **cache_metrics
            ),
            alerts=alert_counts
        )
        
        logger.info(
            "dashboard_completed",
            request_id=request_id,
            response_time_ms=round((time.time() - start_time) * 1000, 2)
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "dashboard_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve dashboard metrics",
                "request_id": request_id
            }
        )


@router.get(
    "/dashboard/health",
    summary="Get Health Dashboard",
    description="""
    Get health status dashboard.
    
    Returns:
    - Overall system health
    - Component health status
    - Recent health checks
    - Degraded/unhealthy components
    """,
    response_model=HealthStatus,
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def get_health_dashboard(
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> HealthStatus:
    """
    Get health status dashboard.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        health_status = await health_checker.get_detailed_health()
        
        response = HealthStatus(
            status=health_status.get("status", "unknown"),
            timestamp=datetime.now(),
            components=health_status.get("components", {}),
            version=settings.version,
            environment=settings.environment.value
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "health_dashboard_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve health dashboard",
                "request_id": request_id
            }
        )


# ============================================================================
# LOGS ENDPOINTS
# ============================================================================

@router.get(
    "/logs",
    summary="Get Logs",
    description="""
    Query and retrieve system logs.
    
    Supports:
    - Time range filtering
    - Log level filtering
    - Service/component filtering
    - Request ID lookup
    - User ID filtering
    - Model ID filtering
    - Full-text search
    - Pagination with cursor
    
    Returns structured log entries in reverse chronological order.
    """,
    response_model=LogsResponse,
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def get_logs(
    request: Request,
    level: Optional[str] = Query(None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    service: Optional[str] = Query(None, description="Filter by service name"),
    component: Optional[str] = Query(None, description="Filter by component name"),
    request_id: Optional[str] = Query(None, description="Filter by request ID"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    search: Optional[str] = Query(None, description="Full-text search in log messages"),
    start_time: Optional[datetime] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End time (ISO format)"),
    limit: int = Query(100, description="Number of logs to return", ge=1, le=1000),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    api_key: str = Depends(verify_api_key)
) -> LogsResponse:
    """
    Query and retrieve system logs.
    """
    request_id_val = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "logs_requested",
        request_id=request_id_val,
        level=level,
        service=service,
        limit=limit
    )
    
    try:
        # Set default time range if not provided
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=24)
        
        # Get logs from aggregator
        logs_result = await log_aggregator.query_logs(
            level=level,
            service=service,
            component=component,
            request_id=request_id,
            user_id=user_id,
            model_id=model_id,
            search=search,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            cursor=cursor
        )
        
        response = LogsResponse(
            total=logs_result["total"],
            logs=[
                LogEntry(**log)
                for log in logs_result["logs"]
            ],
            next_cursor=logs_result.get("next_cursor"),
            query={
                "level": level,
                "service": service,
                "component": component,
                "request_id": request_id,
                "user_id": user_id,
                "model_id": model_id,
                "search": search,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "limit": limit
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "logs_failed",
            request_id=request_id_val,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve logs",
                "request_id": request_id_val
            }
        )


@router.get(
    "/logs/export",
    summary="Export Logs",
    description="""
    Export logs in various formats.
    
    Formats supported:
    - JSON (default)
    - CSV
    - Plain text
    
    Admin only endpoint - may be resource intensive.
    """,
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=5, period=60))]
)
async def export_logs(
    request: Request,
    format: str = Query("json", description="Export format (json, csv, txt)"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    limit: int = Query(10000, description="Maximum logs to export", ge=1, le=100000),
    api_key: str = Depends(verify_admin)
):
    """
    Export logs in various formats.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "logs_export_requested",
        request_id=request_id,
        format=format,
        limit=limit,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        # Set default time range
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        # Get logs
        logs_result = await log_aggregator.query_logs(
            level=level,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # Format based on export type
        if format == "json":
            content = json.dumps(logs_result["logs"], indent=2, default=str)
            media_type = "application/json"
            filename = f"logs_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.json"
            
        elif format == "csv":
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Write headers
            if logs_result["logs"]:
                writer.writerow(logs_result["logs"][0].keys())
                
                # Write rows
                for log in logs_result["logs"]:
                    writer.writerow(log.values())
            
            content = output.getvalue()
            media_type = "text/csv"
            filename = f"logs_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
            
        else:  # txt format
            lines = []
            for log in logs_result["logs"]:
                lines.append(f"[{log['timestamp']}] {log['level']}: {log['message']}")
            
            content = "\n".join(lines)
            media_type = "text/plain"
            filename = f"logs_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.txt"
        
        # Return as downloadable file
        from fastapi.responses import Response
        
        headers = {
            "Content-Disposition": f"attachment; filename={filename}"
        }
        
        logger.info(
            "logs_export_completed",
            request_id=request_id,
            format=format,
            log_count=len(logs_result["logs"])
        )
        
        return Response(
            content=content,
            media_type=media_type,
            headers=headers
        )
        
    except Exception as e:
        logger.error(
            "logs_export_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to export logs",
                "request_id": request_id
            }
        )


# ============================================================================
# ALERTS ENDPOINTS
# ============================================================================

@router.get(
    "/alerts",
    summary="Get Alerts",
    description="""
    Get active and historical alerts.
    
    Returns:
    - Currently firing alerts
    - Acknowledged alerts
    - Resolved alerts
    - Alert history
    
    Supports filtering by severity, status, and time range.
    """,
    response_model=AlertsResponse,
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def get_alerts(
    request: Request,
    severity: Optional[str] = Query(None, description="Filter by severity (info, warning, critical)"),
    status: Optional[str] = Query(None, description="Filter by status (firing, resolved, acknowledged)"),
    service: Optional[str] = Query(None, description="Filter by affected service"),
    model_id: Optional[str] = Query(None, description="Filter by affected model"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    limit: int = Query(100, description="Number of alerts to return", ge=1, le=1000),
    api_key: str = Depends(verify_api_key)
) -> AlertsResponse:
    """
    Get active and historical alerts.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        # Set default time range
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        # Get alerts from manager
        alerts_result = await alert_manager.query_alerts(
            severity=severity,
            status=status,
            service=service,
            model_id=model_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # Get current alert counts
        alert_counts = await alert_manager.get_alert_counts()
        
        response = AlertsResponse(
            total=alerts_result["total"],
            firing=alert_counts.get("firing", 0),
            acknowledged=alert_counts.get("acknowledged", 0),
            alerts=[
                Alert(**alert)
                for alert in alerts_result["alerts"]
            ]
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "alerts_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve alerts",
                "request_id": request_id
            }
        )


@router.post(
    "/alerts/{alert_id}/acknowledge",
    summary="Acknowledge Alert",
    description="""
    Acknowledge an alert to indicate it's being handled.
    
    Acknowledged alerts:
    - Are still firing but marked as acknowledged
    - Won't trigger additional notifications
    - Track who acknowledged them
    
    Admin only endpoint.
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=20, period=60))]
)
async def acknowledge_alert(
    request: Request,
    alert_id: str,
    note: Optional[str] = Query(None, description="Acknowledgment note"),
    api_key: str = Depends(verify_admin)
) -> Dict[str, Any]:
    """
    Acknowledge an alert.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "alert_acknowledged",
        request_id=request_id,
        alert_id=alert_id,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        success = await alert_manager.acknowledge_alert(
            alert_id=alert_id,
            acknowledged_by=getattr(request.state, "user_id", "unknown"),
            note=note
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "AlertNotFound",
                    "message": f"Alert '{alert_id}' not found",
                    "request_id": request_id
                }
            )
        
        return {
            "status": "acknowledged",
            "alert_id": alert_id,
            "acknowledged_by": getattr(request.state, "user_id", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "note": note,
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "alert_acknowledge_failed",
            request_id=request_id,
            alert_id=alert_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to acknowledge alert '{alert_id}'",
                "request_id": request_id
            }
        )


@router.post(
    "/alerts/{alert_id}/resolve",
    summary="Resolve Alert",
    description="""
    Resolve an alert.
    
    Resolved alerts:
    - Are no longer firing
    - Marked as resolved with timestamp
    - Track who resolved them
    
    Admin only endpoint.
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=20, period=60))]
)
async def resolve_alert(
    request: Request,
    alert_id: str,
    note: Optional[str] = Query(None, description="Resolution note"),
    api_key: str = Depends(verify_admin)
) -> Dict[str, Any]:
    """
    Resolve an alert.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "alert_resolved",
        request_id=request_id,
        alert_id=alert_id,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        success = await alert_manager.resolve_alert(
            alert_id=alert_id,
            resolved_by=getattr(request.state, "user_id", "unknown"),
            note=note
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "AlertNotFound",
                    "message": f"Alert '{alert_id}' not found",
                    "request_id": request_id
                }
            )
        
        return {
            "status": "resolved",
            "alert_id": alert_id,
            "resolved_by": getattr(request.state, "user_id", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "note": note,
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "alert_resolve_failed",
            request_id=request_id,
            alert_id=alert_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to resolve alert '{alert_id}'",
                "request_id": request_id
            }
        )


# ============================================================================
# TRACING ENDPOINTS
# ============================================================================

@router.get(
    "/traces/{trace_id}",
    summary="Get Trace",
    description="""
    Get distributed trace by ID.
    
    Returns complete trace information:
    - All spans in the trace
    - Duration of each span
    - Service/component names
    - Error information
    - Metadata
    
    Admin only endpoint - requires tracing enabled.
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=30, period=60))]
)
async def get_trace(
    request: Request,
    trace_id: str,
    api_key: str = Depends(verify_admin)
) -> Dict[str, Any]:
    """
    Get distributed trace by ID.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    if not settings.enable_tracing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "TracingDisabled",
                "message": "Distributed tracing is not enabled",
                "request_id": request_id
            }
        )
    
    logger.info(
        "trace_requested",
        request_id=request_id,
        trace_id=trace_id
    )
    
    try:
        # TODO: Implement trace retrieval from Jaeger/OpenTelemetry
        trace = await metrics_collector.get_trace(trace_id)
        
        if not trace:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "TraceNotFound",
                    "message": f"Trace '{trace_id}' not found",
                    "request_id": request_id
                }
            )
        
        return {
            "trace_id": trace_id,
            "spans": trace.get("spans", []),
            "duration_ms": trace.get("duration_ms"),
            "services": trace.get("services", []),
            "errors": trace.get("errors", []),
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "trace_failed",
            request_id=request_id,
            trace_id=trace_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to retrieve trace '{trace_id}'",
                "request_id": request_id
            }
        )


# ============================================================================
# WEBSOCKET ENDPOINTS FOR REAL-TIME MONITORING
# ============================================================================

@router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for real-time metrics streaming.
    
    Sends live metrics updates every second:
    - System metrics (CPU, memory, GPU)
    - Request rate and latency
    - Error rates
    - Cache hit rate
    - Model performance
    - Active alerts
    
    Authentication via API key in query string or headers.
    """
    await websocket.accept()
    
    client_host = websocket.client.host if websocket.client else "unknown"
    request_id = str(uuid.uuid4())
    
    logger.info(
        "websocket_metrics_connected",
        request_id=request_id,
        client=client_host
    )
    
    try:
        # Authenticate the WebSocket connection
        api_key = websocket.query_params.get("api_key") or websocket.headers.get("X-API-Key")
        
        if not api_key:
            logger.warning(
                "websocket_metrics_auth_failed",
                request_id=request_id,
                client=client_host,
                reason="No API key provided"
            )
            await websocket.close(code=1008, reason="API key required")
            return
        
        from core.security import verify_api_key
        user = await verify_api_key(api_key)
        
        if not user:
            logger.warning(
                "websocket_metrics_auth_failed",
                request_id=request_id,
                client=client_host,
                reason="Invalid API key"
            )
            await websocket.close(code=1008, reason="Invalid API key")
            return
        
        # Send initial message
        await websocket.send_json({
            "type": "connected",
            "message": "Real-time metrics streaming started",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Stream metrics every second
        while True:
            # Get live metrics
            metrics = await metrics_collector.get_live_metrics()
            
            # Get active alerts
            alerts = await alert_manager.get_firing_alerts(limit=5)
            
            # Send to client
            await websocket.send_json({
                "type": "metrics_update",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "alerts": alerts
            })
            
            # Wait 1 second
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        logger.info(
            "websocket_metrics_disconnected",
            request_id=request_id,
            client=client_host
        )
    except Exception as e:
        logger.error(
            "websocket_metrics_error",
            request_id=request_id,
            client=client_host,
            error=str(e),
            exc_info=True
        )
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """
    WebSocket endpoint for real-time log streaming.
    
    Streams new log entries as they happen:
    - All log levels
    - Structured JSON format
    - Real-time filtering by level/service
    
    Admin only - requires admin API key.
    """
    await websocket.accept()
    
    client_host = websocket.client.host if websocket.client else "unknown"
    request_id = str(uuid.uuid4())
    
    logger.info(
        "websocket_logs_connected",
        request_id=request_id,
        client=client_host
    )
    
    try:
        # Authenticate (admin only)
        api_key = websocket.query_params.get("api_key") or websocket.headers.get("X-API-Key")
        
        if not api_key:
            await websocket.close(code=1008, reason="API key required")
            return
        
        from core.security import verify_admin
        admin = await verify_admin(api_key)
        
        if not admin:
            await websocket.close(code=1008, reason="Admin access required")
            return
        
        # Get log level filter if provided
        log_level = websocket.query_params.get("level", "INFO")
        
        # Send initial message
        await websocket.send_json({
            "type": "connected",
            "message": f"Real-time log streaming started (level: {log_level})",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Stream logs in real-time
        async for log_entry in log_aggregator.stream_logs(level=log_level):
            await websocket.send_json({
                "type": "log_entry",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "log": log_entry
            })
            
    except WebSocketDisconnect:
        logger.info(
            "websocket_logs_disconnected",
            request_id=request_id,
            client=client_host
        )
    except Exception as e:
        logger.error(
            "websocket_logs_error",
            request_id=request_id,
            client=client_host,
            error=str(e),
            exc_info=True
        )
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alerts.
    
    Streams alerts as they are triggered:
    - New alert notifications
    - Alert status changes
    - Acknowledgment updates
    - Resolution events
    
    Admin only - requires admin API key.
    """
    await websocket.accept()
    
    client_host = websocket.client.host if websocket.client else "unknown"
    request_id = str(uuid.uuid4())
    
    logger.info(
        "websocket_alerts_connected",
        request_id=request_id,
        client=client_host
    )
    
    try:
        # Authenticate (admin only)
        api_key = websocket.query_params.get("api_key") or websocket.headers.get("X-API-Key")
        
        if not api_key:
            await websocket.close(code=1008, reason="API key required")
            return
        
        from core.security import verify_admin
        admin = await verify_admin(api_key)
        
        if not admin:
            await websocket.close(code=1008, reason="Admin access required")
            return
        
        # Send initial message
        await websocket.send_json({
            "type": "connected",
            "message": "Real-time alert streaming started",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Stream alerts in real-time
        async for alert in alert_manager.stream_alerts():
            await websocket.send_json({
                "type": "alert",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "alert": alert
            })
            
    except WebSocketDisconnect:
        logger.info(
            "websocket_alerts_disconnected",
            request_id=request_id,
            client=client_host
        )
    except Exception as e:
        logger.error(
            "websocket_alerts_error",
            request_id=request_id,
            client=client_host,
            error=str(e),
            exc_info=True
        )
    finally:
        try:
            await websocket.close()
        except:
            pass


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "router",
    "MetricPoint",
    "TimeSeriesResponse",
    "SystemMetricsResponse",
    "RequestMetricsResponse",
    "ModelMetricsResponse",
    "CacheMetricsResponse",
    "LogEntry",
    "LogsResponse",
    "Alert",
    "AlertsResponse",
    "DashboardResponse",
    "HealthStatus"
]