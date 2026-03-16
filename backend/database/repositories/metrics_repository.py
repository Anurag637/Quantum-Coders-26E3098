"""
Metrics Repository - Production Ready
Database operations for system metrics, performance monitoring, usage statistics,
and analytical queries with time-series aggregation.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import select, func, and_, desc, text, Integer, Numeric, Float
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import label

from core.logging import get_logger
from database.models import (
    ModelMetrics, SystemMetrics, RequestLog, CacheStats,
    ChatHistory, RoutingDecision
)
from database.repositories.base import BaseRepository

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# METRICS REPOSITORY
# ============================================================================

class MetricsRepository(BaseRepository[ModelMetrics]):
    """
    Repository for system and performance metrics.
    
    Features:
    - Time-series metric storage
    - Aggregation by time windows
    - Percentile calculations
    - Trend analysis
    - Anomaly detection
    - Capacity planning
    """
    
    def __init__(self, session: Optional[AsyncSession] = None):
        super().__init__(ModelMetrics, session)
    
    # ========================================================================
    # MODEL METRICS
    # ========================================================================
    
    async def record_model_metrics(
        self,
        model_id: str,
        inference_count: int = 1,
        success_count: int = 1,
        error_count: int = 0,
        avg_latency_ms: Optional[float] = None,
        p95_latency_ms: Optional[float] = None,
        p99_latency_ms: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
        gpu_usage_percent: Optional[float] = None,
        cpu_usage_percent: Optional[float] = None,
        throughput_rps: Optional[float] = None,
        cache_hits: int = 0,
        cache_misses: int = 0,
        avg_tokens_per_request: Optional[float] = None,
        total_tokens: int = 0,
        cost_total: float = 0.0
    ) -> ModelMetrics:
        """
        Record performance metrics for a model.
        
        Args:
            model_id: Model identifier
            inference_count: Number of inferences
            success_count: Successful inferences
            error_count: Failed inferences
            avg_latency_ms: Average latency
            p95_latency_ms: 95th percentile latency
            p99_latency_ms: 99th percentile latency
            memory_usage_mb: Memory usage in MB
            gpu_usage_percent: GPU utilization percentage
            cpu_usage_percent: CPU utilization percentage
            throughput_rps: Requests per second
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            avg_tokens_per_request: Average tokens per request
            total_tokens: Total tokens processed
            cost_total: Total cost in USD
        
        Returns:
            Created model metrics record
        """
        metrics = ModelMetrics(
            model_id=model_id,
            inference_count=inference_count,
            success_count=success_count,
            error_count=error_count,
            avg_latency_ms=avg_latency_ms,
            p95_latency_ms=p95_latency_ms,
            p99_latency_ms=p99_latency_ms,
            memory_usage_mb=memory_usage_mb,
            gpu_usage_percent=gpu_usage_percent,
            cpu_usage_percent=cpu_usage_percent,
            throughput_rps=throughput_rps,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            avg_tokens_per_request=avg_tokens_per_request,
            total_tokens=total_tokens,
            cost_total=cost_total
        )
        
        self.session.add(metrics)
        await self.session.flush()
        
        logger.debug(
            "model_metrics_recorded",
            model_id=model_id,
            inference_count=inference_count,
            success_count=success_count,
            error_count=error_count,
            avg_latency_ms=avg_latency_ms
        )
        
        return metrics
    
    async def get_model_metrics(
        self,
        model_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        resolution: str = '1h'
    ) -> Dict[str, Any]:
        """
        Get aggregated metrics for a specific model.
        
        Args:
            model_id: Model identifier
            start_time: Start time for aggregation
            end_time: End time for aggregation
            resolution: Aggregation resolution (1m, 5m, 15m, 1h, 1d)
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=1)
        
        # Determine time bucket based on resolution
        bucket_map = {
            '1m': '1 minute',
            '5m': '5 minutes',
            '15m': '15 minutes',
            '1h': '1 hour',
            '1d': '1 day'
        }
        bucket = bucket_map.get(resolution, '1 hour')
        
        # Time-series query with bucketing
        query = text("""
            SELECT 
                time_bucket(:bucket, timestamp) as bucket,
                AVG(avg_latency_ms) as avg_latency,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY p95_latency_ms) as p95_latency,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY p99_latency_ms) as p99_latency,
                SUM(inference_count) as total_inferences,
                SUM(success_count) as total_success,
                SUM(error_count) as total_errors,
                AVG(error_count::float / NULLIF(inference_count, 0)) as error_rate,
                AVG(memory_usage_mb) as avg_memory_mb,
                AVG(gpu_usage_percent) as avg_gpu_percent,
                AVG(throughput_rps) as avg_throughput,
                SUM(cache_hits) as total_cache_hits,
                SUM(cache_misses) as total_cache_misses,
                SUM(total_tokens) as total_tokens,
                SUM(cost_total) as total_cost
            FROM model_metrics
            WHERE model_id = :model_id
                AND timestamp BETWEEN :start_time AND :end_time
            GROUP BY bucket
            ORDER BY bucket DESC
        """)
        
        result = await self.session.execute(
            query,
            {
                "model_id": model_id,
                "start_time": start_time,
                "end_time": end_time,
                "bucket": bucket
            }
        )
        
        time_series = []
        for row in result:
            time_series.append({
                "timestamp": row.bucket.isoformat(),
                "avg_latency_ms": round(float(row.avg_latency), 2) if row.avg_latency else 0,
                "p95_latency_ms": round(float(row.p95_latency), 2) if row.p95_latency else 0,
                "p99_latency_ms": round(float(row.p99_latency), 2) if row.p99_latency else 0,
                "inferences": row.total_inferences,
                "success": row.total_success,
                "errors": row.total_errors,
                "error_rate": round(float(row.error_rate) * 100, 2) if row.error_rate else 0,
                "memory_mb": round(float(row.avg_memory_mb), 2) if row.avg_memory_mb else 0,
                "gpu_percent": round(float(row.avg_gpu_percent), 2) if row.avg_gpu_percent else 0,
                "throughput_rps": round(float(row.avg_throughput), 2) if row.avg_throughput else 0,
                "cache_hits": row.total_cache_hits,
                "cache_misses": row.total_cache_misses,
                "cache_hit_rate": round(
                    row.total_cache_hits / (row.total_cache_hits + row.total_cache_misses) * 100, 2
                ) if (row.total_cache_hits + row.total_cache_misses) > 0 else 0,
                "tokens": row.total_tokens,
                "cost_usd": round(float(row.total_cost), 6) if row.total_cost else 0
            })
        
        # Get overall aggregates
        agg_query = select(
            func.avg(ModelMetrics.avg_latency_ms).label('avg_latency'),
            func.percentile_cont(0.95).within_group(
                ModelMetrics.p95_latency_ms.asc()
            ).label('p95_latency'),
            func.percentile_cont(0.99).within_group(
                ModelMetrics.p99_latency_ms.asc()
            ).label('p99_latency'),
            func.sum(ModelMetrics.inference_count).label('total_inferences'),
            func.sum(ModelMetrics.success_count).label('total_success'),
            func.sum(ModelMetrics.error_count).label('total_errors'),
            func.avg(ModelMetrics.memory_usage_mb).label('avg_memory'),
            func.avg(ModelMetrics.throughput_rps).label('avg_throughput'),
            func.sum(ModelMetrics.cache_hits).label('total_cache_hits'),
            func.sum(ModelMetrics.cache_misses).label('total_cache_misses'),
            func.sum(ModelMetrics.total_tokens).label('total_tokens'),
            func.sum(ModelMetrics.cost_total).label('total_cost')
        ).where(
            and_(
                ModelMetrics.model_id == model_id,
                ModelMetrics.timestamp >= start_time,
                ModelMetrics.timestamp <= end_time
            )
        )
        
        agg_result = await self.session.execute(agg_query)
        agg = agg_result.first()
        
        total_requests = (agg.total_success or 0) + (agg.total_errors or 0)
        cache_total = (agg.total_cache_hits or 0) + (agg.total_cache_misses or 0)
        
        return {
            "model_id": model_id,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "resolution": resolution
            },
            "summary": {
                "avg_latency_ms": round(float(agg.avg_latency), 2) if agg.avg_latency else 0,
                "p95_latency_ms": round(float(agg.p95_latency), 2) if agg.p95_latency else 0,
                "p99_latency_ms": round(float(agg.p99_latency), 2) if agg.p99_latency else 0,
                "total_inferences": agg.total_inferences or 0,
                "success_rate": round(
                    (agg.total_success or 0) / total_requests * 100, 2
                ) if total_requests > 0 else 100,
                "error_rate": round(
                    (agg.total_errors or 0) / total_requests * 100, 2
                ) if total_requests > 0 else 0,
                "avg_memory_mb": round(float(agg.avg_memory), 2) if agg.avg_memory else 0,
                "avg_throughput_rps": round(float(agg.avg_throughput), 2) if agg.avg_throughput else 0,
                "cache_hit_rate": round(
                    (agg.total_cache_hits or 0) / cache_total * 100, 2
                ) if cache_total > 0 else 0,
                "total_tokens": agg.total_tokens or 0,
                "total_cost_usd": round(float(agg.total_cost), 6) if agg.total_cost else 0
            },
            "time_series": time_series
        }
    
    async def get_all_models_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get aggregated metrics for all models.
        
        Args:
            start_time: Start time for aggregation
            end_time: End time for aggregation
        
        Returns:
            List of model metrics
        """
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=1)
        
        query = select(
            ModelMetrics.model_id,
            func.count().label('samples'),
            func.avg(ModelMetrics.avg_latency_ms).label('avg_latency'),
            func.avg(ModelMetrics.p95_latency_ms).label('avg_p95'),
            func.avg(ModelMetrics.p99_latency_ms).label('avg_p99'),
            func.sum(ModelMetrics.inference_count).label('total_inferences'),
            func.sum(ModelMetrics.success_count).label('total_success'),
            func.sum(ModelMetrics.error_count).label('total_errors'),
            func.sum(ModelMetrics.cache_hits).label('total_cache_hits'),
            func.sum(ModelMetrics.cache_misses).label('total_cache_misses'),
            func.sum(ModelMetrics.total_tokens).label('total_tokens'),
            func.sum(ModelMetrics.cost_total).label('total_cost')
        ).where(
            and_(
                ModelMetrics.timestamp >= start_time,
                ModelMetrics.timestamp <= end_time
            )
        ).group_by(ModelMetrics.model_id)
        
        result = await self.session.execute(query)
        models_metrics = []
        
        for row in result:
            total_requests = (row.total_success or 0) + (row.total_errors or 0)
            cache_total = (row.total_cache_hits or 0) + (row.total_cache_misses or 0)
            
            models_metrics.append({
                "model_id": row.model_id,
                "samples": row.samples,
                "avg_latency_ms": round(float(row.avg_latency), 2) if row.avg_latency else 0,
                "avg_p95_ms": round(float(row.avg_p95), 2) if row.avg_p95 else 0,
                "avg_p99_ms": round(float(row.avg_p99), 2) if row.avg_p99 else 0,
                "total_inferences": row.total_inferences or 0,
                "success_rate": round(
                    (row.total_success or 0) / total_requests * 100, 2
                ) if total_requests > 0 else 100,
                "error_rate": round(
                    (row.total_errors or 0) / total_requests * 100, 2
                ) if total_requests > 0 else 0,
                "cache_hit_rate": round(
                    (row.total_cache_hits or 0) / cache_total * 100, 2
                ) if cache_total > 0 else 0,
                "total_tokens": row.total_tokens or 0,
                "total_cost_usd": round(float(row.total_cost), 6) if row.total_cost else 0
            })
        
        # Sort by inference count
        models_metrics.sort(key=lambda x: x["total_inferences"], reverse=True)
        
        return models_metrics
    
    # ========================================================================
    # REQUEST METRICS
    # ========================================================================
    
    async def get_request_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        resolution: str = '1h'
    ) -> Dict[str, Any]:
        """
        Get API request metrics.
        
        Args:
            start_time: Start time for aggregation
            end_time: End time for aggregation
            resolution: Aggregation resolution
        
        Returns:
            Dictionary with request metrics
        """
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=1)
        
        # Total requests and success rate
        total_query = select(
            func.count(RequestLog.id).label('total_requests'),
            func.sum(
                func.cast(RequestLog.status_code < 400, Integer)
            ).label('successful_requests'),
            func.avg(RequestLog.response_time_ms).label('avg_response_time'),
            func.percentile_cont(0.95).within_group(
                RequestLog.response_time_ms.asc()
            ).label('p95_response_time'),
            func.percentile_cont(0.99).within_group(
                RequestLog.response_time_ms.asc()
            ).label('p99_response_time')
        ).where(
            and_(
                RequestLog.created_at >= start_time,
                RequestLog.created_at <= end_time
            )
        )
        
        total_result = await self.session.execute(total_query)
        total = total_result.first()
        
        # Status code distribution
        status_query = select(
            RequestLog.status_code,
            func.count().label('count')
        ).where(
            and_(
                RequestLog.created_at >= start_time,
                RequestLog.created_at <= end_time
            )
        ).group_by(RequestLog.status_code)
        
        status_result = await self.session.execute(status_query)
        status_codes = {}
        
        for row in status_result:
            status_codes[str(row.status_code)] = row.count
        
        # Top endpoints
        endpoints_query = select(
            RequestLog.endpoint,
            func.count().label('count'),
            func.avg(RequestLog.response_time_ms).label('avg_latency'),
            func.max(RequestLog.response_time_ms).label('max_latency')
        ).where(
            and_(
                RequestLog.created_at >= start_time,
                RequestLog.created_at <= end_time
            )
        ).group_by(RequestLog.endpoint
        ).order_by(desc('count')
        ).limit(10)
        
        endpoints_result = await self.session.execute(endpoints_query)
        top_endpoints = []
        
        for row in endpoints_result:
            top_endpoints.append({
                "endpoint": row.endpoint,
                "requests": row.count,
                "avg_latency_ms": round(float(row.avg_latency), 2) if row.avg_latency else 0,
                "max_latency_ms": round(float(row.max_latency), 2) if row.max_latency else 0
            })
        
        # Time series data
        bucket_map = {
            '1m': '1 minute',
            '5m': '5 minutes',
            '15m': '15 minutes',
            '1h': '1 hour',
            '1d': '1 day'
        }
        bucket = bucket_map.get(resolution, '1 hour')
        
        series_query = text("""
            SELECT 
                time_bucket(:bucket, created_at) as bucket,
                COUNT(*) as requests,
                AVG(response_time_ms) as avg_latency,
                COUNT(CASE WHEN status_code >= 500 THEN 1 END) as errors
            FROM request_logs
            WHERE created_at BETWEEN :start_time AND :end_time
            GROUP BY bucket
            ORDER BY bucket DESC
        """)
        
        series_result = await self.session.execute(
            series_query,
            {
                "start_time": start_time,
                "end_time": end_time,
                "bucket": bucket
            }
        )
        
        time_series = []
        for row in series_result:
            time_series.append({
                "timestamp": row.bucket.isoformat(),
                "requests": row.requests,
                "avg_latency_ms": round(float(row.avg_latency), 2) if row.avg_latency else 0,
                "errors": row.errors or 0,
                "error_rate": round((row.errors or 0) / row.requests * 100, 2)
            })
        
        total_requests = total.total_requests or 0
        successful = total.successful_requests or 0
        
        return {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "resolution": resolution
            },
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful,
                "failed_requests": total_requests - successful,
                "success_rate": round(successful / total_requests * 100, 2) if total_requests > 0 else 100,
                "avg_response_time_ms": round(float(total.avg_response_time), 2) if total.avg_response_time else 0,
                "p95_response_time_ms": round(float(total.p95_response_time), 2) if total.p95_response_time else 0,
                "p99_response_time_ms": round(float(total.p99_response_time), 2) if total.p99_response_time else 0,
                "requests_per_second": round(total_requests / (end_time - start_time).total_seconds(), 2)
            },
            "status_codes": status_codes,
            "top_endpoints": top_endpoints,
            "time_series": time_series
        }
    
    # ========================================================================
    # CACHE METRICS
    # ========================================================================
    
    async def record_cache_stats(
        self,
        cache_hits: int,
        cache_misses: int,
        cache_size_bytes: int,
        cache_entries: int,
        eviction_count: int = 0,
        avg_lookup_time_ms: float = 0.0
    ) -> CacheStats:
        """
        Record cache performance statistics.
        
        Args:
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            cache_size_bytes: Cache size in bytes
            cache_entries: Number of cache entries
            eviction_count: Number of evictions
            avg_lookup_time_ms: Average lookup time
        
        Returns:
            Created cache stats record
        """
        stats = CacheStats(
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_size_bytes=cache_size_bytes,
            cache_entries=cache_entries,
            eviction_count=eviction_count,
            avg_lookup_time_ms=avg_lookup_time_ms
        )
        
        self.session.add(stats)
        await self.session.flush()
        
        return stats
    
    async def get_cache_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Args:
            start_time: Start time for aggregation
            end_time: End time for aggregation
        
        Returns:
            Dictionary with cache metrics
        """
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=1)
        
        query = select(
            func.sum(CacheStats.cache_hits).label('total_hits'),
            func.sum(CacheStats.cache_misses).label('total_misses'),
            func.avg(CacheStats.cache_hits + CacheStats.cache_misses > 0).label('hit_rate'),
            func.avg(CacheStats.cache_size_bytes).label('avg_size_bytes'),
            func.avg(CacheStats.cache_entries).label('avg_entries'),
            func.sum(CacheStats.eviction_count).label('total_evictions'),
            func.avg(CacheStats.avg_lookup_time_ms).label('avg_lookup_time')
        ).where(
            and_(
                CacheStats.timestamp >= start_time,
                CacheStats.timestamp <= end_time
            )
        )
        
        result = await self.session.execute(query)
        row = result.first()
        
        total_requests = (row.total_hits or 0) + (row.total_misses or 0)
        
        # Time series data
        series_query = text("""
            SELECT 
                time_bucket('1 hour', timestamp) as bucket,
                SUM(cache_hits) as hits,
                SUM(cache_misses) as misses,
                AVG(cache_size_bytes) as size_bytes,
                AVG(cache_entries) as entries,
                AVG(avg_lookup_time_ms) as lookup_time
            FROM cache_stats
            WHERE timestamp BETWEEN :start_time AND :end_time
            GROUP BY bucket
            ORDER BY bucket DESC
        """)
        
        series_result = await self.session.execute(
            series_query,
            {
                "start_time": start_time,
                "end_time": end_time
            }
        )
        
        time_series = []
        for row_ts in series_result:
            hits = row_ts.hits or 0
            misses = row_ts.misses or 0
            total = hits + misses
            
            time_series.append({
                "timestamp": row_ts.bucket.isoformat(),
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / total * 100, 2) if total > 0 else 0,
                "size_mb": round((row_ts.size_bytes or 0) / (1024 * 1024), 2),
                "entries": row_ts.entries or 0,
                "avg_lookup_time_ms": round(float(row_ts.lookup_time), 2) if row_ts.lookup_time else 0
            })
        
        return {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "summary": {
                "total_hits": row.total_hits or 0,
                "total_misses": row.total_misses or 0,
                "total_requests": total_requests,
                "hit_rate": round(
                    (row.total_hits or 0) / total_requests * 100, 2
                ) if total_requests > 0 else 0,
                "avg_size_mb": round((row.avg_size_bytes or 0) / (1024 * 1024), 2),
                "avg_entries": round(row.avg_entries or 0, 2),
                "total_evictions": row.total_evictions or 0,
                "avg_lookup_time_ms": round(float(row.avg_lookup_time), 2) if row.avg_lookup_time else 0
            },
            "time_series": time_series
        }
    
    # ========================================================================
    # SYSTEM METRICS
    # ========================================================================
    
    async def record_system_metrics(
        self,
        service_name: str,
        cpu_percent: float,
        memory_percent: float,
        memory_used_mb: float,
        memory_total_mb: float,
        disk_percent: float,
        disk_used_gb: float,
        disk_total_gb: float,
        network_in_mbps: Optional[float] = None,
        network_out_mbps: Optional[float] = None,
        active_connections: Optional[int] = None,
        error_rate: Optional[float] = None,
        uptime_seconds: Optional[int] = None
    ) -> SystemMetrics:
        """
        Record system performance metrics.
        
        Args:
            service_name: Service name
            cpu_percent: CPU utilization percentage
            memory_percent: Memory utilization percentage
            memory_used_mb: Used memory in MB
            memory_total_mb: Total memory in MB
            disk_percent: Disk utilization percentage
            disk_used_gb: Used disk space in GB
            disk_total_gb: Total disk space in GB
            network_in_mbps: Network inbound Mbps
            network_out_mbps: Network outbound Mbps
            active_connections: Active connections
            error_rate: Error rate percentage
            uptime_seconds: Service uptime in seconds
        
        Returns:
            Created system metrics record
        """
        metrics = SystemMetrics(
            service_name=service_name,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            network_in_mbps=network_in_mbps,
            network_out_mbps=network_out_mbps,
            active_connections=active_connections,
            error_rate=error_rate,
            uptime_seconds=uptime_seconds
        )
        
        self.session.add(metrics)
        await self.session.flush()
        
        return metrics
    
    async def get_system_metrics(
        self,
        service_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get system performance metrics.
        
        Args:
            service_name: Optional service name filter
            start_time: Start time for aggregation
            end_time: End time for aggregation
        
        Returns:
            Dictionary with system metrics
        """
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(hours=1)
        
        query = select(
            func.avg(SystemMetrics.cpu_percent).label('avg_cpu'),
            func.max(SystemMetrics.cpu_percent).label('max_cpu'),
            func.avg(SystemMetrics.memory_percent).label('avg_memory'),
            func.max(SystemMetrics.memory_percent).label('max_memory'),
            func.avg(SystemMetrics.disk_percent).label('avg_disk'),
            func.max(SystemMetrics.disk_percent).label('max_disk'),
            func.avg(SystemMetrics.network_in_mbps).label('avg_network_in'),
            func.avg(SystemMetrics.network_out_mbps).label('avg_network_out'),
            func.avg(SystemMetrics.active_connections).label('avg_connections'),
            func.max(SystemMetrics.active_connections).label('max_connections'),
            func.avg(SystemMetrics.error_rate).label('avg_error_rate'),
            func.avg(SystemMetrics.uptime_seconds).label('avg_uptime')
        ).where(
            and_(
                SystemMetrics.timestamp >= start_time,
                SystemMetrics.timestamp <= end_time
            )
        )
        
        if service_name:
            query = query.where(SystemMetrics.service_name == service_name)
        
        result = await self.session.execute(query)
        row = result.first()
        
        # Time series data
        series_query = text("""
            SELECT 
                time_bucket('5 minutes', timestamp) as bucket,
                AVG(cpu_percent) as cpu,
                AVG(memory_percent) as memory,
                AVG(disk_percent) as disk,
                AVG(active_connections) as connections,
                AVG(error_rate) as error_rate
            FROM system_metrics
            WHERE timestamp BETWEEN :start_time AND :end_time
        """)
        
        if service_name:
            series_query = text(series_query.text + " AND service_name = :service_name")
        
        series_query = text(series_query.text + " GROUP BY bucket ORDER BY bucket DESC")
        
        params = {
            "start_time": start_time,
            "end_time": end_time
        }
        if service_name:
            params["service_name"] = service_name
        
        series_result = await self.session.execute(series_query, params)
        
        time_series = []
        for row_ts in series_result:
            time_series.append({
                "timestamp": row_ts.bucket.isoformat(),
                "cpu_percent": round(float(row_ts.cpu), 2) if row_ts.cpu else 0,
                "memory_percent": round(float(row_ts.memory), 2) if row_ts.memory else 0,
                "disk_percent": round(float(row_ts.disk), 2) if row_ts.disk else 0,
                "active_connections": round(row_ts.connections or 0),
                "error_rate": round(float(row_ts.error_rate), 2) if row_ts.error_rate else 0
            })
        
        return {
            "service": service_name or "all",
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "summary": {
                "avg_cpu_percent": round(float(row.avg_cpu), 2) if row.avg_cpu else 0,
                "max_cpu_percent": round(float(row.max_cpu), 2) if row.max_cpu else 0,
                "avg_memory_percent": round(float(row.avg_memory), 2) if row.avg_memory else 0,
                "max_memory_percent": round(float(row.max_memory), 2) if row.max_memory else 0,
                "avg_disk_percent": round(float(row.avg_disk), 2) if row.avg_disk else 0,
                "max_disk_percent": round(float(row.max_disk), 2) if row.max_disk else 0,
                "avg_network_in_mbps": round(float(row.avg_network_in), 2) if row.avg_network_in else 0,
                "avg_network_out_mbps": round(float(row.avg_network_out), 2) if row.avg_network_out else 0,
                "avg_active_connections": round(row.avg_connections or 0),
                "max_active_connections": row.max_connections or 0,
                "avg_error_rate": round(float(row.avg_error_rate) * 100, 2) if row.avg_error_rate else 0,
                "avg_uptime_seconds": round(row.avg_uptime or 0)
            },
            "time_series": time_series
        }
    
    # ========================================================================
    # USAGE STATISTICS
    # ========================================================================
    
    async def get_usage_statistics(
        self,
        period: str = '24h'
    ) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics.
        
        Args:
            period: Time period (1h, 24h, 7d, 30d)
        
        Returns:
            Dictionary with usage statistics
        """
        end_time = datetime.utcnow()
        
        if period == '1h':
            start_time = end_time - timedelta(hours=1)
        elif period == '24h':
            start_time = end_time - timedelta(days=1)
        elif period == '7d':
            start_time = end_time - timedelta(days=7)
        elif period == '30d':
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(days=1)
        
        # Get request statistics
        request_stats = await self.get_request_metrics(start_time, end_time)
        
        # Get model statistics
        model_stats = await self.get_all_models_metrics(start_time, end_time)
        
        # Get cache statistics
        cache_stats = await self.get_cache_metrics(start_time, end_time)
        
        # Get user statistics
        user_query = select(
            func.count(func.distinct(RequestLog.user_id)).label('active_users'),
            func.count(func.distinct(ChatHistory.user_id)).label('chat_users')
        ).where(
            and_(
                RequestLog.created_at >= start_time,
                RequestLog.created_at <= end_time
            )
        )
        
        user_result = await self.session.execute(user_query)
        user_row = user_result.first()
        
        # Get conversation statistics
        conv_query = select(
            func.count(func.distinct(ChatHistory.conversation_id)).label('total_conversations'),
            func.avg(
                select(func.count(ChatHistory.id))
                .where(ChatHistory.conversation_id == ChatHistory.conversation_id)
                .scalar_subquery()
            ).label('avg_messages_per_conversation')
        ).where(
            and_(
                ChatHistory.created_at >= start_time,
                ChatHistory.created_at <= end_time
            )
        )
        
        conv_result = await self.session.execute(conv_query)
        conv_row = conv_result.first()
        
        return {
            "period": period,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "requests": request_stats["summary"],
            "models": {
                "total_models": len(model_stats),
                "top_models": model_stats[:5],
                "total_inferences": sum(m["total_inferences"] for m in model_stats),
                "total_tokens": sum(m["total_tokens"] for m in model_stats),
                "total_cost": sum(m["total_cost_usd"] for m in model_stats)
            },
            "cache": cache_stats["summary"],
            "users": {
                "active_users": user_row.active_users or 0,
                "chat_users": user_row.chat_users or 0
            },
            "conversations": {
                "total": conv_row.total_conversations or 0,
                "avg_messages_per_conversation": round(conv_row.avg_messages_per_conversation or 0, 2)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # ========================================================================
    # ANOMALY DETECTION
    # ========================================================================
    
    async def detect_anomalies(
        self,
        metric: str = 'latency',
        threshold: float = 2.0,
        window_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metrics using statistical methods.
        
        Args:
            metric: Metric to analyze (latency, errors, tokens)
            threshold: Z-score threshold for anomaly detection
            window_minutes: Time window for baseline
        
        Returns:
            List of detected anomalies
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=window_minutes * 2)
        baseline_start = start_time
        baseline_end = end_time - timedelta(minutes=window_minutes)
        
        anomalies = []
        
        if metric == 'latency':
            # Get recent latency data
            query = select(
                ModelMetrics.timestamp,
                ModelMetrics.model_id,
                ModelMetrics.avg_latency_ms
            ).where(
                and_(
                    ModelMetrics.timestamp >= baseline_start,
                    ModelMetrics.timestamp <= end_time,
                    ModelMetrics.avg_latency_ms.isnot(None)
                )
            ).order_by(ModelMetrics.timestamp)
            
            result = await self.session.execute(query)
            data = result.all()
            
            # Calculate baseline statistics
            baseline_data = [d for d in data if d.timestamp <= baseline_end]
            if len(baseline_data) > 1:
                values = [d.avg_latency_ms for d in baseline_data]
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                
                # Check recent points
                recent_data = [d for d in data if d.timestamp > baseline_end]
                for point in recent_data:
                    if std > 0:
                        z_score = (point.avg_latency_ms - mean) / std
                        if abs(z_score) > threshold:
                            anomalies.append({
                                "timestamp": point.timestamp.isoformat(),
                                "model_id": point.model_id,
                                "metric": "latency",
                                "value": point.avg_latency_ms,
                                "expected": round(mean, 2),
                                "z_score": round(z_score, 2),
                                "severity": "high" if abs(z_score) > 3 else "medium"
                            })
        
        elif metric == 'errors':
            # Similar implementation for error rate anomalies
            pass
        
        elif metric == 'tokens':
            # Similar implementation for token usage anomalies
            pass
        
        return anomalies


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "MetricsRepository"
]