"""
Model Repository - Production Ready
Database operations for model management, including CRUD operations,
performance metrics, usage tracking, and model registry management.
"""

import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sqlalchemy import select, update, delete, and_, or_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import text

from core.logging import get_logger
from database.models import ModelRegistry, ModelMetrics, User
from database.session import get_session
from config import settings

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# MODEL REPOSITORY
# ============================================================================

class ModelRepository:
    """
    Repository for model-related database operations.
    
    Features:
    - CRUD operations for model registry
    - Model performance metrics
    - Usage tracking and analytics
    - Health status monitoring
    - Model capability queries
    - Provider-based filtering
    """
    
    def __init__(self):
        self.session = get_session()
        logger.info("model_repository_initialized")
    
    # ========================================================================
    # MODEL REGISTRY CRUD
    # ========================================================================
    
    async def create_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new model entry in the registry.
        
        Args:
            model_data: Model configuration and metadata
        
        Returns:
            Created model data
        """
        async with self.session() as session:
            try:
                # Generate UUID for new model
                model_id = model_data.get("id") or str(uuid.uuid4())
                
                # Create model instance
                model = ModelRegistry(
                    id=model_id,
                    model_id=model_data.get("model_id", model_id),
                    model_name=model_data["name"],
                    provider=model_data.get("provider", "unknown"),
                    model_type=model_data.get("type", "unknown"),
                    library=model_data.get("library", "transformers"),
                    format=model_data.get("format"),
                    quantization=model_data.get("quantization"),
                    context_size=model_data.get("context_size", 2048),
                    status=model_data.get("status", "pending"),
                    capabilities=model_data.get("capabilities", []),
                    api_endpoint=model_data.get("api_endpoint"),
                    api_key_env=model_data.get("api_key_env"),
                    requires_api_key=model_data.get("requires_api_key", False),
                    file_size_mb=model_data.get("file_size_mb"),
                    memory_required_gb=model_data.get("memory_required_gb"),
                    gpu_memory_required_gb=model_data.get("gpu_memory_required_gb"),
                    is_loaded=False,
                    load_count=0,
                    error_count=0,
                    metadata=model_data.get("metadata", {}),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                session.add(model)
                await session.commit()
                await session.refresh(model)
                
                logger.info(
                    "model_created",
                    model_id=model.model_id,
                    provider=model.provider,
                    type=model.model_type
                )
                
                return self._model_to_dict(model)
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to create model: {e}")
                raise
    
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model by ID.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Model data or None if not found
        """
        async with self.session() as session:
            try:
                query = select(ModelRegistry).where(
                    or_(
                        ModelRegistry.id == model_id,
                        ModelRegistry.model_id == model_id
                    )
                )
                
                result = await session.execute(query)
                model = result.scalar_one_or_none()
                
                if model:
                    return self._model_to_dict(model)
                
                return None
                
            except Exception as e:
                logger.error(f"Failed to get model {model_id}: {e}")
                raise
    
    async def get_model_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """
        Get models by provider.
        
        Args:
            provider: Model provider name
        
        Returns:
            List of models from the provider
        """
        async with self.session() as session:
            try:
                query = select(ModelRegistry).where(
                    ModelRegistry.provider.ilike(f"%{provider}%")
                ).order_by(ModelRegistry.model_name)
                
                result = await session.execute(query)
                models = result.scalars().all()
                
                return [self._model_to_dict(model) for model in models]
                
            except Exception as e:
                logger.error(f"Failed to get models for provider {provider}: {e}")
                raise
    
    async def get_all_models(
        self,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
        provider: Optional[str] = None,
        capability: Optional[str] = None,
        is_loaded: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get all models with optional filtering.
        
        Args:
            status: Filter by status
            model_type: Filter by model type
            provider: Filter by provider
            capability: Filter by capability
            is_loaded: Filter by load status
            limit: Pagination limit
            offset: Pagination offset
        
        Returns:
            Tuple of (models list, total count)
        """
        async with self.session() as session:
            try:
                # Build query
                query = select(ModelRegistry)
                count_query = select(func.count()).select_from(ModelRegistry)
                
                # Apply filters
                filters = []
                if status:
                    filters.append(ModelRegistry.status == status)
                if model_type:
                    filters.append(ModelRegistry.model_type == model_type)
                if provider:
                    filters.append(ModelRegistry.provider.ilike(f"%{provider}%"))
                if capability:
                    filters.append(ModelRegistry.capabilities.contains([capability]))
                if is_loaded is not None:
                    filters.append(ModelRegistry.is_loaded == is_loaded)
                
                if filters:
                    query = query.where(and_(*filters))
                    count_query = count_query.where(and_(*filters))
                
                # Get total count
                count_result = await session.execute(count_query)
                total = count_result.scalar()
                
                # Apply pagination
                query = query.order_by(
                    desc(ModelRegistry.is_loaded),
                    ModelRegistry.model_name
                ).offset(offset).limit(limit)
                
                result = await session.execute(query)
                models = result.scalars().all()
                
                return [self._model_to_dict(model) for model in models], total
                
            except Exception as e:
                logger.error(f"Failed to get all models: {e}")
                raise
    
    async def update_model(
        self,
        model_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update model information.
        
        Args:
            model_id: Model identifier
            updates: Fields to update
        
        Returns:
            Updated model data or None if not found
        """
        async with self.session() as session:
            try:
                # Check if model exists
                query = select(ModelRegistry).where(
                    or_(
                        ModelRegistry.id == model_id,
                        ModelRegistry.model_id == model_id
                    )
                )
                result = await session.execute(query)
                model = result.scalar_one_or_none()
                
                if not model:
                    return None
                
                # Update fields
                for key, value in updates.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                
                model.updated_at = datetime.utcnow()
                
                await session.commit()
                await session.refresh(model)
                
                logger.info(
                    "model_updated",
                    model_id=model.model_id,
                    updates=list(updates.keys())
                )
                
                return self._model_to_dict(model)
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to update model {model_id}: {e}")
                raise
    
    async def delete_model(self, model_id: str) -> bool:
        """
        Delete model from registry.
        
        Args:
            model_id: Model identifier
        
        Returns:
            True if deleted, False if not found
        """
        async with self.session() as session:
            try:
                query = delete(ModelRegistry).where(
                    or_(
                        ModelRegistry.id == model_id,
                        ModelRegistry.model_id == model_id
                    )
                )
                
                result = await session.execute(query)
                await session.commit()
                
                if result.rowcount > 0:
                    logger.info("model_deleted", model_id=model_id)
                    return True
                
                return False
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to delete model {model_id}: {e}")
                raise
    
    # ========================================================================
    # MODEL STATUS MANAGEMENT
    # ========================================================================
    
    async def update_model_status(
        self,
        model_id: str,
        status: str,
        is_loaded: Optional[bool] = None,
        loaded_at: Optional[datetime] = None,
        error: Optional[str] = None,
        memory_usage_mb: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update model loading status.
        
        Args:
            model_id: Model identifier
            status: New status (pending, loading, ready, error, unloaded)
            is_loaded: Whether model is loaded in memory
            loaded_at: When model was loaded
            error: Error message if status is error
            memory_usage_mb: Current memory usage in MB
        
        Returns:
            Updated model data
        """
        async with self.session() as session:
            try:
                # Get model
                query = select(ModelRegistry).where(
                    or_(
                        ModelRegistry.id == model_id,
                        ModelRegistry.model_id == model_id
                    )
                )
                result = await session.execute(query)
                model = result.scalar_one_or_none()
                
                if not model:
                    return None
                
                # Update status
                model.status = status
                model.updated_at = datetime.utcnow()
                
                if is_loaded is not None:
                    model.is_loaded = is_loaded
                
                if loaded_at:
                    model.loaded_at = loaded_at
                
                if error:
                    model.last_error = error
                    model.error_count = (model.error_count or 0) + 1
                
                if memory_usage_mb is not None:
                    # Store memory usage in metadata
                    metadata = model.metadata or {}
                    metadata["memory_usage_mb"] = memory_usage_mb
                    model.metadata = metadata
                
                await session.commit()
                await session.refresh(model)
                
                logger.info(
                    "model_status_updated",
                    model_id=model.model_id,
                    status=status,
                    is_loaded=is_loaded
                )
                
                return self._model_to_dict(model)
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to update model status {model_id}: {e}")
                raise
    
    async def increment_load_count(self, model_id: str) -> Optional[int]:
        """
        Increment model load counter.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Updated load count or None if model not found
        """
        async with self.session() as session:
            try:
                query = select(ModelRegistry).where(
                    or_(
                        ModelRegistry.id == model_id,
                        ModelRegistry.model_id == model_id
                    )
                )
                result = await session.execute(query)
                model = result.scalar_one_or_none()
                
                if not model:
                    return None
                
                model.load_count = (model.load_count or 0) + 1
                model.updated_at = datetime.utcnow()
                
                await session.commit()
                
                return model.load_count
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to increment load count for {model_id}: {e}")
                raise
    
    async def get_models_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get all models with a specific status.
        
        Args:
            status: Model status
        
        Returns:
            List of models with the status
        """
        async with self.session() as session:
            try:
                query = select(ModelRegistry).where(
                    ModelRegistry.status == status
                ).order_by(ModelRegistry.model_name)
                
                result = await session.execute(query)
                models = result.scalars().all()
                
                return [self._model_to_dict(model) for model in models]
                
            except Exception as e:
                logger.error(f"Failed to get models by status {status}: {e}")
                raise
    
    # ========================================================================
    # MODEL METRICS
    # ========================================================================
    
    async def record_metrics(
        self,
        model_id: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Record performance metrics for a model.
        
        Args:
            model_id: Model identifier
            metrics: Performance metrics
        
        Returns:
            Created metrics record
        """
        async with self.session() as session:
            try:
                # Get model to ensure it exists
                model = await self.get_model(model_id)
                if not model:
                    raise ValueError(f"Model {model_id} not found")
                
                # Create metrics record
                model_metrics = ModelMetrics(
                    id=str(uuid.uuid4()),
                    model_id=model_id,
                    timestamp=datetime.utcnow(),
                    inference_count=metrics.get("inference_count", 1),
                    success_count=metrics.get("success_count", 1),
                    error_count=metrics.get("error_count", 0),
                    avg_latency_ms=metrics.get("avg_latency_ms", 0),
                    p95_latency_ms=metrics.get("p95_latency_ms", 0),
                    p99_latency_ms=metrics.get("p99_latency_ms", 0),
                    memory_usage_mb=metrics.get("memory_usage_mb"),
                    gpu_usage_percent=metrics.get("gpu_usage_percent"),
                    cpu_usage_percent=metrics.get("cpu_usage_percent"),
                    throughput_rps=metrics.get("throughput_rps"),
                    cache_hits=metrics.get("cache_hits", 0),
                    cache_misses=metrics.get("cache_misses", 0),
                    avg_tokens_per_request=metrics.get("avg_tokens_per_request"),
                    total_tokens=metrics.get("total_tokens", 0),
                    cost_total=metrics.get("cost_total", 0)
                )
                
                session.add(model_metrics)
                await session.commit()
                await session.refresh(model_metrics)
                
                logger.debug(
                    "model_metrics_recorded",
                    model_id=model_id,
                    inference_count=model_metrics.inference_count,
                    latency_ms=model_metrics.avg_latency_ms
                )
                
                return self._metrics_to_dict(model_metrics)
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to record metrics for {model_id}: {e}")
                raise
    
    async def get_model_metrics(
        self,
        model_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: Model identifier
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of records
        
        Returns:
            List of metrics records
        """
        async with self.session() as session:
            try:
                query = select(ModelMetrics).where(
                    ModelMetrics.model_id == model_id
                )
                
                if start_time:
                    query = query.where(ModelMetrics.timestamp >= start_time)
                if end_time:
                    query = query.where(ModelMetrics.timestamp <= end_time)
                
                query = query.order_by(
                    desc(ModelMetrics.timestamp)
                ).limit(limit)
                
                result = await session.execute(query)
                metrics = result.scalars().all()
                
                return [self._metrics_to_dict(m) for m in metrics]
                
            except Exception as e:
                logger.error(f"Failed to get metrics for {model_id}: {e}")
                raise
    
    async def get_model_metrics_aggregated(
        self,
        model_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get aggregated metrics for a model over time period.
        
        Args:
            model_id: Model identifier
            hours: Number of hours to aggregate
        
        Returns:
            Aggregated metrics
        """
        async with self.session() as session:
            try:
                cutoff = datetime.utcnow() - timedelta(hours=hours)
                
                # Get metrics in time range
                query = select(ModelMetrics).where(
                    and_(
                        ModelMetrics.model_id == model_id,
                        ModelMetrics.timestamp >= cutoff
                    )
                )
                
                result = await session.execute(query)
                metrics = result.scalars().all()
                
                if not metrics:
                    return {
                        "model_id": model_id,
                        "period_hours": hours,
                        "total_requests": 0,
                        "success_count": 0,
                        "error_count": 0,
                        "avg_latency_ms": 0,
                        "p95_latency_ms": 0,
                        "p99_latency_ms": 0,
                        "total_tokens": 0,
                        "avg_tokens_per_request": 0,
                        "cache_hit_rate": 0,
                        "cost_total": 0
                    }
                
                # Calculate aggregates
                total_requests = sum(m.inference_count for m in metrics)
                total_success = sum(m.success_count for m in metrics)
                total_errors = sum(m.error_count for m in metrics)
                total_tokens = sum(m.total_tokens for m in metrics)
                total_cost = sum(m.cost_total for m in metrics)
                
                # Weighted averages
                weighted_latency = sum(
                    m.avg_latency_ms * m.inference_count for m in metrics
                ) / total_requests if total_requests > 0 else 0
                
                # Get p95 and p99 from most recent record
                latest = metrics[0] if metrics else None
                
                # Cache hit rate
                total_cache = sum(m.cache_hits + m.cache_misses for m in metrics)
                cache_hit_rate = (
                    sum(m.cache_hits for m in metrics) / total_cache
                    if total_cache > 0 else 0
                )
                
                return {
                    "model_id": model_id,
                    "period_hours": hours,
                    "total_requests": total_requests,
                    "success_count": total_success,
                    "error_count": total_errors,
                    "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0,
                    "avg_latency_ms": round(weighted_latency, 2),
                    "p95_latency_ms": latest.p95_latency_ms if latest else 0,
                    "p99_latency_ms": latest.p99_latency_ms if latest else 0,
                    "total_tokens": total_tokens,
                    "avg_tokens_per_request": total_tokens / total_requests if total_requests > 0 else 0,
                    "cache_hits": sum(m.cache_hits for m in metrics),
                    "cache_misses": sum(m.cache_misses for m in metrics),
                    "cache_hit_rate": round(cache_hit_rate * 100, 2),
                    "cost_total": round(total_cost, 6),
                    "cost_per_request": round(total_cost / total_requests, 8) if total_requests > 0 else 0
                }
                
            except Exception as e:
                logger.error(f"Failed to get aggregated metrics for {model_id}: {e}")
                raise
    
    # ========================================================================
    # MODEL ANALYTICS
    # ========================================================================
    
    async def get_model_usage_stats(self) -> Dict[str, Any]:
        """
        Get overall model usage statistics.
        
        Returns:
            Aggregated usage statistics across all models
        """
        async with self.session() as session:
            try:
                # Total models by status
                status_query = select(
                    ModelRegistry.status,
                    func.count().label('count')
                ).group_by(ModelRegistry.status)
                
                status_result = await session.execute(status_query)
                status_counts = {row[0]: row[1] for row in status_result}
                
                # Models by type
                type_query = select(
                    ModelRegistry.model_type,
                    func.count().label('count')
                ).group_by(ModelRegistry.model_type)
                
                type_result = await session.execute(type_query)
                type_counts = {row[0]: row[1] for row in type_result}
                
                # Models by provider
                provider_query = select(
                    ModelRegistry.provider,
                    func.count().label('count')
                ).group_by(ModelRegistry.provider)
                
                provider_result = await session.execute(provider_query)
                provider_counts = {row[0]: row[1] for row in provider_result}
                
                # Loaded models count
                loaded_query = select(func.count()).where(
                    ModelRegistry.is_loaded == True
                )
                loaded_result = await session.execute(loaded_query)
                loaded_count = loaded_result.scalar()
                
                # Error models count
                error_query = select(func.count()).where(
                    ModelRegistry.status == 'error'
                )
                error_result = await session.execute(error_query)
                error_count = error_result.scalar()
                
                # Total inference count across all models
                inference_query = select(func.sum(ModelMetrics.inference_count))
                inference_result = await session.execute(inference_query)
                total_inferences = inference_result.scalar() or 0
                
                return {
                    "total_models": sum(status_counts.values()),
                    "by_status": status_counts,
                    "by_type": type_counts,
                    "by_provider": provider_counts,
                    "loaded_count": loaded_count,
                    "error_count": error_count,
                    "total_inferences": total_inferences,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Failed to get model usage stats: {e}")
                raise
    
    async def get_top_models(
        self,
        metric: str = "inference_count",
        limit: int = 10,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get top performing models by various metrics.
        
        Args:
            metric: Metric to rank by (inference_count, avg_latency_ms, cost_total)
            limit: Number of models to return
            hours: Time window in hours
        
        Returns:
            List of top models with metrics
        """
        async with self.session() as session:
            try:
                cutoff = datetime.utcnow() - timedelta(hours=hours)
                
                # Subquery to get aggregated metrics per model
                subquery = select(
                    ModelMetrics.model_id,
                    func.sum(ModelMetrics.inference_count).label('total_inferences'),
                    func.avg(ModelMetrics.avg_latency_ms).label('avg_latency'),
                    func.sum(ModelMetrics.total_tokens).label('total_tokens'),
                    func.sum(ModelMetrics.cost_total).label('total_cost'),
                    func.avg(ModelMetrics.cache_hits * 1.0 / 
                            func.nullif(ModelMetrics.cache_hits + ModelMetrics.cache_misses, 0)
                    ).label('cache_hit_rate')
                ).where(
                    ModelMetrics.timestamp >= cutoff
                ).group_by(
                    ModelMetrics.model_id
                )
                
                # Order by selected metric
                if metric == "inference_count":
                    subquery = subquery.order_by(desc('total_inferences'))
                elif metric == "latency":
                    subquery = subquery.order_by('avg_latency')
                elif metric == "cost":
                    subquery = subquery.order_by(desc('total_cost'))
                
                subquery = subquery.limit(limit)
                
                result = await session.execute(subquery)
                metrics_rows = result.all()
                
                # Get model details for each top model
                top_models = []
                for row in metrics_rows:
                    model = await self.get_model(row.model_id)
                    if model:
                        top_models.append({
                            **model,
                            "metrics": {
                                "total_inferences": row.total_inferences,
                                "avg_latency_ms": round(row.avg_latency, 2) if row.avg_latency else 0,
                                "total_tokens": row.total_tokens,
                                "total_cost": round(row.total_cost, 6) if row.total_cost else 0,
                                "cache_hit_rate": round(row.cache_hit_rate * 100, 2) if row.cache_hit_rate else 0
                            }
                        })
                
                return top_models
                
            except Exception as e:
                logger.error(f"Failed to get top models: {e}")
                raise
    
    # ========================================================================
    # MODEL CAPABILITY QUERIES
    # ========================================================================
    
    async def get_models_by_capability(
        self,
        capability: str,
        only_loaded: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get models that have a specific capability.
        
        Args:
            capability: Capability to filter by
            only_loaded: Only return loaded models
        
        Returns:
            List of models with the capability
        """
        async with self.session() as session:
            try:
                query = select(ModelRegistry).where(
                    ModelRegistry.capabilities.contains([capability])
                )
                
                if only_loaded:
                    query = query.where(ModelRegistry.is_loaded == True)
                
                query = query.order_by(ModelRegistry.model_name)
                
                result = await session.execute(query)
                models = result.scalars().all()
                
                return [self._model_to_dict(model) for model in models]
                
            except Exception as e:
                logger.error(f"Failed to get models by capability {capability}: {e}")
                raise
    
    async def get_all_capabilities(self) -> List[str]:
        """
        Get all unique capabilities across all models.
        
        Returns:
            List of all capabilities
        """
        async with self.session() as session:
            try:
                # This query uses PostgreSQL's unnest and array_agg
                query = text("""
                    SELECT DISTINCT unnest(capabilities) as capability
                    FROM model_registry
                    ORDER BY capability
                """)
                
                result = await session.execute(query)
                capabilities = [row[0] for row in result]
                
                return capabilities
                
            except Exception as e:
                logger.error(f"Failed to get all capabilities: {e}")
                raise
    
    async def get_models_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """
        Get models by provider name.
        
        Args:
            provider: Provider name (partial match)
        
        Returns:
            List of models from the provider
        """
        async with self.session() as session:
            try:
                query = select(ModelRegistry).where(
                    ModelRegistry.provider.ilike(f"%{provider}%")
                ).order_by(ModelRegistry.model_name)
                
                result = await session.execute(query)
                models = result.scalars().all()
                
                return [self._model_to_dict(model) for model in models]
                
            except Exception as e:
                logger.error(f"Failed to get models by provider {provider}: {e}")
                raise
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _model_to_dict(self, model: ModelRegistry) -> Dict[str, Any]:
        """Convert model ORM object to dictionary."""
        return {
            "id": model.id,
            "model_id": model.model_id,
            "name": model.model_name,
            "provider": model.provider,
            "type": model.model_type,
            "library": model.library,
            "format": model.format,
            "quantization": model.quantization,
            "context_size": model.context_size,
            "status": model.status,
            "capabilities": model.capabilities,
            "api_endpoint": model.api_endpoint,
            "api_key_env": model.api_key_env,
            "requires_api_key": model.requires_api_key,
            "file_size_mb": model.file_size_mb,
            "memory_required_gb": model.memory_required_gb,
            "gpu_memory_required_gb": model.gpu_memory_required_gb,
            "is_loaded": model.is_loaded,
            "load_count": model.load_count,
            "error_count": model.error_count,
            "last_loaded": model.last_loaded.isoformat() if model.last_loaded else None,
            "last_error": model.last_error,
            "metadata": model.metadata,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None
        }
    
    def _metrics_to_dict(self, metrics: ModelMetrics) -> Dict[str, Any]:
        """Convert metrics ORM object to dictionary."""
        return {
            "id": metrics.id,
            "model_id": metrics.model_id,
            "timestamp": metrics.timestamp.isoformat() if metrics.timestamp else None,
            "inference_count": metrics.inference_count,
            "success_count": metrics.success_count,
            "error_count": metrics.error_count,
            "avg_latency_ms": metrics.avg_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "p99_latency_ms": metrics.p99_latency_ms,
            "memory_usage_mb": metrics.memory_usage_mb,
            "gpu_usage_percent": metrics.gpu_usage_percent,
            "cpu_usage_percent": metrics.cpu_usage_percent,
            "throughput_rps": metrics.throughput_rps,
            "cache_hits": metrics.cache_hits,
            "cache_misses": metrics.cache_misses,
            "avg_tokens_per_request": metrics.avg_tokens_per_request,
            "total_tokens": metrics.total_tokens,
            "cost_total": metrics.cost_total
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_model_repository = None


def get_model_repository() -> ModelRepository:
    """Get singleton model repository instance."""
    global _model_repository
    if not _model_repository:
        _model_repository = ModelRepository()
    return _model_repository


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ModelRepository",
    "get_model_repository"
]