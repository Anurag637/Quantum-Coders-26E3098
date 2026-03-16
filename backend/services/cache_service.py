"""
Cache Service - Production Ready
Orchestrates multi-level caching with Redis, semantic similarity,
intelligent invalidation, and comprehensive monitoring.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, timedelta
from uuid import uuid4

from core.logging import get_logger, get_cache_logger
from core.exceptions import CacheError
from config import settings
from cache.cache_manager import CacheManager, get_cache_manager
from cache.redis_cache import RedisCache, get_redis_cache
from cache.semantic_cache import SemanticCache, get_semantic_cache
from monitoring.metrics import MetricsCollector
from database.repositories.metrics_repository import MetricsRepository
from utils.token_counter import TokenCounter
from utils.cost_calculator import CostCalculator

# Initialize loggers
logger = get_logger(__name__)
cache_logger = get_cache_logger()

# ============================================================================
# CACHE STRATEGIES
# ============================================================================

class CacheStrategy:
    """Base class for cache strategies."""
    
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache."""
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        raise NotImplementedError
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries."""
        raise NotImplementedError


class RedisCacheStrategy(CacheStrategy):
    """Redis-based cache strategy."""
    
    def __init__(self, redis_cache: RedisCache):
        self.redis = redis_cache
    
    async def get(self, key: str, default: Any = None) -> Optional[Any]:
        value = await self.redis.get(key)
        if value:
            try:
                return json.loads(value.decode('utf-8'))
            except:
                return value
        return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        if not isinstance(value, (str, bytes)):
            value = json.dumps(value)
        return await self.redis.set(key, value, ttl=ttl)
    
    async def delete(self, key: str) -> bool:
        return await self.redis.delete(key) > 0
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        if pattern:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
        return 0


class SemanticCacheStrategy(CacheStrategy):
    """Semantic similarity-based cache strategy."""
    
    def __init__(self, semantic_cache: SemanticCache):
        self.semantic = semantic_cache
    
    async def get(
        self,
        key: str,
        default: Any = None,
        threshold: Optional[float] = None
    ) -> Optional[Any]:
        # Key is actually the prompt for semantic cache
        result = await self.semantic.get(
            prompt=key,
            threshold=threshold or settings.cache.similarity_threshold
        )
        if result.get("hit"):
            return result
        return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        # key is prompt, value is response
        success = await self.semantic.set(
            prompt=key,
            response=value.get("response", value),
            model=value.get("model", "unknown"),
            ttl=ttl,
            tokens_saved=value.get("tokens_saved", 0),
            cost_saved_usd=value.get("cost_saved_usd", 0.0)
        )
        return success
    
    async def delete(self, key: str) -> bool:
        return await self.semantic.delete(key)
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        return await self.semantic.clear()


# ============================================================================
# CACHE SERVICE
# ============================================================================

class CacheService:
    """
    High-level cache orchestration service.
    
    Features:
    - Multi-level caching (L1 memory, L2 Redis, L3 semantic)
    - Automatic strategy selection
    - Cache warming
    - Intelligent invalidation
    - Cache statistics aggregation
    - Performance monitoring
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        redis_cache: Optional[RedisCache] = None,
        semantic_cache: Optional[SemanticCache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        metrics_repository: Optional[MetricsRepository] = None
    ):
        # Initialize cache backends
        self.cache_manager = cache_manager or get_cache_manager()
        self.redis_cache = redis_cache or get_redis_cache()
        self.semantic_cache = semantic_cache or get_semantic_cache()
        
        # Initialize strategies
        self.redis_strategy = RedisCacheStrategy(self.redis_cache)
        self.semantic_strategy = SemanticCacheStrategy(self.semantic_cache)
        
        # Monitoring
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.metrics_repository = metrics_repository or MetricsRepository()
        
        # Token counter for cache savings
        self.token_counter = TokenCounter()
        self.cost_calculator = CostCalculator()
        
        # Cache warming
        self.warm_tasks: Dict[str, asyncio.Task] = {}
        self.warm_schedules: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            "total_operations": 0,
            "total_hits": 0,
            "total_misses": 0,
            "total_semantic_hits": 0,
            "total_bytes_saved": 0,
            "total_tokens_saved": 0,
            "total_cost_saved_usd": 0.0,
            "avg_latency_ms": 0,
            "cache_size_bytes": 0,
            "cache_entries": 0,
            "semantic_cache_entries": 0,
            "last_warm_time": None
        }
        
        logger.info(
            "cache_service_initialized",
            cache_manager_ready=bool(cache_manager),
            redis_ready=bool(redis_cache),
            semantic_ready=bool(semantic_cache)
        )
    
    # ========================================================================
    # CORE OPERATIONS
    # ========================================================================
    
    async def get(
        self,
        key: str,
        default: Any = None,
        use_redis: bool = True,
        use_semantic: bool = True,
        semantic_threshold: Optional[float] = None,
        record_stats: bool = True
    ) -> Optional[Any]:
        """
        Get value from cache with multi-level fallback.
        
        Strategy:
        1. Try L1 cache (memory) - fastest
        2. Try Redis cache - fast
        3. Try semantic cache - slower but intelligent
        4. Return default
        
        Args:
            key: Cache key (prompt for semantic cache)
            default: Default value if not found
            use_redis: Whether to check Redis
            use_semantic: Whether to check semantic cache
            semantic_threshold: Similarity threshold for semantic cache
            record_stats: Whether to record statistics
        
        Returns:
            Cached value or default
        """
        start_time = time.time()
        request_id = str(uuid4())[:8]
        
        if record_stats:
            self.stats["total_operations"] += 1
        
        logger.debug(
            "cache_get_requested",
            request_id=request_id,
            key=key[:50] + "..." if len(key) > 50 else key,
            use_redis=use_redis,
            use_semantic=use_semantic
        )
        
        # Level 1: L1 Cache (Memory)
        l1_result = await self.cache_manager.get(key, use_l1=True)
        if l1_result is not None:
            latency = (time.time() - start_time) * 1000
            
            if record_stats:
                self.stats["total_hits"] += 1
                self.stats["avg_latency_ms"] = (
                    (self.stats["avg_latency_ms"] * (self.stats["total_hits"] - 1) + latency) /
                    self.stats["total_hits"]
                )
            
            logger.debug(
                "cache_l1_hit",
                request_id=request_id,
                key=key[:50] + "...",
                latency_ms=round(latency, 2)
            )
            
            await self.metrics_collector.record_cache_hit(
                cache_type="l1",
                latency_ms=latency,
                key=key
            )
            
            return l1_result
        
        # Level 2: Redis Cache
        if use_redis:
            redis_result = await self.redis_strategy.get(key)
            if redis_result is not None:
                latency = (time.time() - start_time) * 1000
                
                # Update L1 cache
                await self.cache_manager.set(key, redis_result, ttl=60)  # 1 minute L1 TTL
                
                if record_stats:
                    self.stats["total_hits"] += 1
                    self.stats["avg_latency_ms"] = (
                        (self.stats["avg_latency_ms"] * (self.stats["total_hits"] - 1) + latency) /
                        self.stats["total_hits"]
                    )
                
                logger.debug(
                    "cache_redis_hit",
                    request_id=request_id,
                    key=key[:50] + "...",
                    latency_ms=round(latency, 2)
                )
                
                await self.metrics_collector.record_cache_hit(
                    cache_type="redis",
                    latency_ms=latency,
                    key=key
                )
                
                return redis_result
        
        # Level 3: Semantic Cache
        if use_semantic:
            semantic_result = await self.semantic_strategy.get(
                key,
                threshold=semantic_threshold
            )
            
            if semantic_result and semantic_result.get("hit"):
                latency = (time.time() - start_time) * 1000
                
                if record_stats:
                    self.stats["total_hits"] += 1
                    self.stats["total_semantic_hits"] += 1
                    self.stats["avg_latency_ms"] = (
                        (self.stats["avg_latency_ms"] * (self.stats["total_hits"] - 1) + latency) /
                        self.stats["total_hits"]
                    )
                    
                    # Track savings
                    tokens_saved = semantic_result.get("tokens_saved", 0)
                    cost_saved = semantic_result.get("cost_saved_usd", 0)
                    self.stats["total_tokens_saved"] += tokens_saved
                    self.stats["total_cost_saved_usd"] += cost_saved
                
                logger.debug(
                    "cache_semantic_hit",
                    request_id=request_id,
                    key=key[:50] + "...",
                    similarity=semantic_result.get("similarity", 0),
                    latency_ms=round(latency, 2),
                    tokens_saved=tokens_saved if record_stats else 0,
                    cost_saved_usd=round(cost_saved, 6) if record_stats else 0
                )
                
                await self.metrics_collector.record_cache_hit(
                    cache_type="semantic",
                    latency_ms=latency,
                    similarity=semantic_result.get("similarity", 0),
                    tokens_saved=tokens_saved if record_stats else 0,
                    cost_saved_usd=cost_saved if record_stats else 0,
                    key=key
                )
                
                return semantic_result.get("response")
        
        # Cache miss
        latency = (time.time() - start_time) * 1000
        
        if record_stats:
            self.stats["total_misses"] += 1
        
        logger.debug(
            "cache_miss",
            request_id=request_id,
            key=key[:50] + "...",
            latency_ms=round(latency, 2)
        )
        
        await self.metrics_collector.record_cache_miss(
            cache_type="all",
            latency_ms=latency,
            key=key
        )
        
        return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        use_redis: bool = True,
        use_semantic: bool = False,
        semantic_metadata: Optional[Dict[str, Any]] = None,
        record_stats: bool = True
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Tags for group invalidation
            use_redis: Whether to store in Redis
            use_semantic: Whether to store in semantic cache
            semantic_metadata: Metadata for semantic cache
            record_stats: Whether to record statistics
        
        Returns:
            True if successful
        """
        start_time = time.time()
        request_id = str(uuid4())[:8]
        
        logger.debug(
            "cache_set_requested",
            request_id=request_id,
            key=key[:50] + "...",
            ttl=ttl,
            tags=tags,
            use_redis=use_redis,
            use_semantic=use_semantic
        )
        
        success = True
        
        # Set in L1 cache
        l1_ttl = min(ttl or settings.cache.default_ttl, 60)  # Max 1 minute for L1
        l1_success = await self.cache_manager.set(
            key,
            value,
            ttl=l1_ttl,
            tags=tags,
            use_l1=True
        )
        
        if not l1_success:
            logger.warning(
                "cache_l1_set_failed",
                request_id=request_id,
                key=key[:50] + "..."
            )
            success = False
        
        # Set in Redis
        if use_redis:
            redis_success = await self.redis_strategy.set(
                key,
                value,
                ttl=ttl,
                tags=tags
            )
            
            if not redis_success:
                logger.warning(
                    "cache_redis_set_failed",
                    request_id=request_id,
                    key=key[:50] + "..."
                )
                success = False
        
        # Set in semantic cache
        if use_semantic:
            semantic_value = {
                "response": value if isinstance(value, str) else value.get("content", value),
                "model": semantic_metadata.get("model", "unknown") if semantic_metadata else "unknown",
                "tokens_saved": semantic_metadata.get("tokens", 0) if semantic_metadata else 0,
                "cost_saved_usd": semantic_metadata.get("cost", 0) if semantic_metadata else 0.0
            }
            
            semantic_success = await self.semantic_strategy.set(
                key,
                semantic_value,
                ttl=ttl,
                tags=tags,
                metadata=semantic_metadata
            )
            
            if not semantic_success:
                logger.warning(
                    "cache_semantic_set_failed",
                    request_id=request_id,
                    key=key[:50] + "..."
                )
                success = False
        
        latency = (time.time() - start_time) * 1000
        
        logger.debug(
            "cache_set_completed",
            request_id=request_id,
            key=key[:50] + "...",
            success=success,
            latency_ms=round(latency, 2)
        )
        
        await self.metrics_collector.record_cache_set(
            success=success,
            latency_ms=latency,
            key=key,
            cache_type="redis" if use_redis else "l1"
        )
        
        return success
    
    async def delete(
        self,
        key: str,
        from_redis: bool = True,
        from_semantic: bool = True
    ) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            from_redis: Delete from Redis
            from_semantic: Delete from semantic cache
        
        Returns:
            True if deleted from any cache
        """
        request_id = str(uuid4())[:8]
        
        logger.debug(
            "cache_delete_requested",
            request_id=request_id,
            key=key[:50] + "...",
            from_redis=from_redis,
            from_semantic=from_semantic
        )
        
        success = False
        
        # Delete from L1 cache
        l1_deleted = await self.cache_manager.delete(key)
        if l1_deleted:
            success = True
        
        # Delete from Redis
        if from_redis:
            redis_deleted = await self.redis_strategy.delete(key)
            if redis_deleted:
                success = True
        
        # Delete from semantic cache
        if from_semantic:
            semantic_deleted = await self.semantic_strategy.delete(key)
            if semantic_deleted:
                success = True
        
        logger.debug(
            "cache_delete_completed",
            request_id=request_id,
            key=key[:50] + "...",
            success=success
        )
        
        return success
    
    async def clear(
        self,
        pattern: Optional[str] = None,
        clear_redis: bool = True,
        clear_semantic: bool = True
    ) -> Dict[str, int]:
        """
        Clear cache entries.
        
        Args:
            pattern: Key pattern to clear
            clear_redis: Clear Redis cache
            clear_semantic: Clear semantic cache
        
        Returns:
            Dictionary with counts per cache type
        """
        request_id = str(uuid4())[:8]
        
        logger.info(
            "cache_clear_requested",
            request_id=request_id,
            pattern=pattern,
            clear_redis=clear_redis,
            clear_semantic=clear_semantic
        )
        
        results = {
            "l1": 0,
            "redis": 0,
            "semantic": 0
        }
        
        # Clear L1 cache
        results["l1"] = await self.cache_manager.clear(pattern)
        
        # Clear Redis
        if clear_redis:
            results["redis"] = await self.redis_strategy.clear(pattern)
        
        # Clear semantic cache
        if clear_semantic:
            results["semantic"] = await self.semantic_strategy.clear()
        
        total = sum(results.values())
        
        logger.info(
            "cache_cleared",
            request_id=request_id,
            pattern=pattern,
            entries_removed=total,
            breakdown=results
        )
        
        return results
    
    # ========================================================================
    # INTELLIGENT CACHING
    # ========================================================================
    
    async def get_or_compute(
        self,
        key: str,
        compute_func: Callable,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        use_semantic: bool = False,
        semantic_threshold: Optional[float] = None,
        force_recompute: bool = False,
        *args,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Get from cache or compute and cache.
        
        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            ttl: Cache TTL
            tags: Cache tags
            use_semantic: Whether to use semantic cache
            semantic_threshold: Similarity threshold
            force_recompute: Skip cache and recompute
            *args: Arguments for compute function
            **kwargs: Keyword arguments for compute function
        
        Returns:
            Tuple of (value, was_cached)
        """
        start_time = time.time()
        request_id = str(uuid4())[:8]
        
        logger.debug(
            "cache_get_or_compute",
            request_id=request_id,
            key=key[:50] + "...",
            force_recompute=force_recompute
        )
        
        # Try cache first
        if not force_recompute:
            cached_value = await self.get(
                key=key,
                use_semantic=use_semantic,
                semantic_threshold=semantic_threshold,
                record_stats=True
            )
            
            if cached_value is not None:
                logger.debug(
                    "cache_hit_skip_compute",
                    request_id=request_id,
                    key=key[:50] + "...",
                    latency_ms=round((time.time() - start_time) * 1000, 2)
                )
                return cached_value, True
        
        # Compute value
        logger.debug(
            "cache_miss_computing",
            request_id=request_id,
            key=key[:50] + "..."
        )
        
        try:
            if asyncio.iscoroutinefunction(compute_func):
                value = await compute_func(*args, **kwargs)
            else:
                value = compute_func(*args, **kwargs)
        except Exception as e:
            logger.error(
                "cache_compute_failed",
                request_id=request_id,
                key=key[:50] + "...",
                error=str(e),
                exc_info=True
            )
            raise
        
        # Cache computed value
        if value is not None:
            await self.set(
                key=key,
                value=value,
                ttl=ttl,
                tags=tags,
                use_semantic=use_semantic,
                record_stats=True
            )
        
        logger.debug(
            "cache_compute_completed",
            request_id=request_id,
            key=key[:50] + "...",
            latency_ms=round((time.time() - start_time) * 1000, 2)
        )
        
        return value, False
    
    # ========================================================================
    # CACHE WARMING
    # ========================================================================
    
    async def warm_cache(
        self,
        name: str,
        keys: List[str],
        value_func: Callable,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        concurrency: int = 5,
        schedule: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Warm up cache with frequently accessed keys.
        
        Args:
            name: Warming task name
            keys: List of keys to warm
            value_func: Function to compute values
            ttl: Cache TTL
            tags: Cache tags
            concurrency: Concurrent requests
            schedule: Cron schedule expression
        
        Returns:
            Dictionary with warming results
        """
        request_id = str(uuid4())[:8]
        
        logger.info(
            "cache_warming_started",
            request_id=request_id,
            name=name,
            key_count=len(keys),
            concurrency=concurrency,
            schedule=schedule
        )
        
        # Cancel existing warming task
        if name in self.warm_tasks:
            self.warm_tasks[name].cancel()
        
        # Store schedule
        if schedule:
            self.warm_schedules[name] = {
                "schedule": schedule,
                "keys": keys,
                "value_func": value_func,
                "ttl": ttl,
                "tags": tags,
                "concurrency": concurrency,
                "last_run": None,
                "next_run": None
            }
        
        # Create warming task
        task = asyncio.create_task(
            self._warm_cache_task(
                request_id=request_id,
                name=name,
                keys=keys,
                value_func=value_func,
                ttl=ttl,
                tags=tags,
                concurrency=concurrency
            )
        )
        
        self.warm_tasks[name] = task
        
        try:
            results = await task
            self.stats["last_warm_time"] = datetime.utcnow().isoformat()
            
            logger.info(
                "cache_warming_completed",
                request_id=request_id,
                name=name,
                successful=results["successful"],
                failed=results["failed"],
                total_time_ms=round(results["total_time_ms"], 2)
            )
            
            return results
            
        except asyncio.CancelledError:
            logger.info(
                "cache_warming_cancelled",
                request_id=request_id,
                name=name
            )
            return {
                "status": "cancelled",
                "name": name,
                "successful": 0,
                "failed": 0
            }
        except Exception as e:
            logger.error(
                "cache_warming_failed",
                request_id=request_id,
                name=name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _warm_cache_task(
        self,
        request_id: str,
        name: str,
        keys: List[str],
        value_func: Callable,
        ttl: Optional[int],
        tags: Optional[List[str]],
        concurrency: int
    ) -> Dict[str, Any]:
        """Background task for cache warming."""
        start_time = time.time()
        successful = 0
        failed = 0
        semaphore = asyncio.Semaphore(concurrency)
        
        async def _warm_key(key: str):
            async with semaphore:
                try:
                    # Compute value
                    if asyncio.iscoroutinefunction(value_func):
                        value = await value_func(key)
                    else:
                        value = value_func(key)
                    
                    # Cache value
                    success = await self.set(
                        key=key,
                        value=value,
                        ttl=ttl,
                        tags=tags,
                        record_stats=False
                    )
                    
                    if success:
                        nonlocal successful
                        successful += 1
                        logger.debug(
                            "cache_key_warmed",
                            request_id=request_id,
                            name=name,
                            key=key[:50] + "..."
                        )
                    else:
                        nonlocal failed
                        failed += 1
                        logger.warning(
                            "cache_key_warm_failed",
                            request_id=request_id,
                            name=name,
                            key=key[:50] + "..."
                        )
                        
                except Exception as e:
                    failed += 1
                    logger.error(
                        "cache_key_warm_error",
                        request_id=request_id,
                        name=name,
                        key=key[:50] + "...",
                        error=str(e)
                    )
        
        # Warm keys concurrently
        tasks = [_warm_key(key) for key in keys]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Update schedule
        if name in self.warm_schedules:
            self.warm_schedules[name]["last_run"] = datetime.utcnow().isoformat()
        
        return {
            "status": "completed",
            "name": name,
            "successful": successful,
            "failed": failed,
            "total": len(keys),
            "total_time_ms": round(total_time_ms, 2),
            "keys_per_second": round(len(keys) / (total_time_ms / 1000), 2) if total_time_ms > 0 else 0
        }
    
    async def stop_warming(self, name: Optional[str] = None):
        """
        Stop cache warming tasks.
        
        Args:
            name: Task name (None for all)
        """
        if name:
            if name in self.warm_tasks:
                self.warm_tasks[name].cancel()
                del self.warm_tasks[name]
            if name in self.warm_schedules:
                del self.warm_schedules[name]
            logger.info(f"Stopped cache warming: {name}")
        else:
            for task_name, task in self.warm_tasks.items():
                task.cancel()
            self.warm_tasks.clear()
            self.warm_schedules.clear()
            logger.info("Stopped all cache warming tasks")
    
    # ========================================================================
    # TAG-BASED OPERATIONS
    # ========================================================================
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate all cache entries with a specific tag.
        
        Args:
            tag: Cache tag
        
        Returns:
            Number of entries invalidated
        """
        request_id = str(uuid4())[:8]
        
        logger.info(
            "cache_tag_invalidation",
            request_id=request_id,
            tag=tag
        )
        
        # Invalidate in Redis
        redis_deleted = await self.redis_cache.delete_by_tag(tag)
        
        # Invalidate in semantic cache
        semantic_deleted = await self.semantic_cache.delete_by_tag(tag)
        
        total = redis_deleted + semantic_deleted
        
        logger.info(
            "cache_tag_invalidated",
            request_id=request_id,
            tag=tag,
            entries_removed=total
        )
        
        return total
    
    async def get_by_tag(self, tag: str) -> List[Any]:
        """
        Get all cache entries with a specific tag.
        
        Args:
            tag: Cache tag
        
        Returns:
            List of cached values
        """
        return await self.redis_cache.get_by_tag(tag)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in any cache layer."""
        return await self.cache_manager.exists(key)
    
    async def ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key."""
        return await self.redis_cache.ttl(key)
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on existing key."""
        return await self.redis_cache.expire(key, ttl)
    
    async def persist(self, key: str) -> bool:
        """Remove expiration from key."""
        return await self.redis_cache.persist(key)
    
    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter."""
        return await self.redis_cache.incr(key, amount)
    
    async def decr(self, key: str, amount: int = 1) -> Optional[int]:
        """Decrement counter."""
        return await self.redis_cache.decr(key, amount)
    
    # ========================================================================
    # STATISTICS AND MONITORING
    # ========================================================================
    
    async def get_stats(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Args:
            detailed: Include detailed per-cache stats
        
        Returns:
            Dictionary with cache statistics
        """
        # Get stats from each cache layer
        cache_manager_stats = await self.cache_manager.get_stats()
        redis_stats = await self.redis_cache.get_stats()
        semantic_stats = await self.semantic_cache.get_stats()
        
        # Calculate hit rate
        total_requests = self.stats["total_hits"] + self.stats["total_misses"]
        hit_rate = (self.stats["total_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate savings
        token_savings = self.stats["total_tokens_saved"]
        cost_savings = self.stats["total_cost_saved_usd"]
        
        # Estimate bytes saved (rough approximation)
        bytes_per_token = 4  # Approximate bytes per token
        bytes_saved = token_savings * bytes_per_token
        self.stats["total_bytes_saved"] = bytes_saved
        
        stats = {
            "summary": {
                "total_operations": self.stats["total_operations"],
                "total_hits": self.stats["total_hits"],
                "total_misses": self.stats["total_misses"],
                "total_semantic_hits": self.stats["total_semantic_hits"],
                "hit_rate": round(hit_rate, 2),
                "avg_latency_ms": round(self.stats["avg_latency_ms"], 2),
                "total_tokens_saved": token_savings,
                "total_cost_saved_usd": round(cost_savings, 6),
                "total_bytes_saved": bytes_saved,
                "cache_size_bytes": cache_manager_stats.get("total_size_bytes", 0),
                "cache_entries": cache_manager_stats.get("total_entries", 0),
                "semantic_cache_entries": semantic_stats.get("cache_size", 0),
                "last_warm_time": self.stats["last_warm_time"]
            }
        }
        
        if detailed:
            stats["detailed"] = {
                "cache_manager": cache_manager_stats,
                "redis": redis_stats,
                "semantic": semantic_stats,
                "warming_tasks": {
                    name: {
                        "status": "running" if task and not task.done() else "completed",
                        "schedule": self.warm_schedules.get(name, {}).get("schedule"),
                        "last_run": self.warm_schedules.get(name, {}).get("last_run")
                    }
                    for name, task in self.warm_tasks.items()
                }
            }
        
        return stats
    
    async def reset_stats(self):
        """Reset cache statistics."""
        self.stats = {
            "total_operations": 0,
            "total_hits": 0,
            "total_misses": 0,
            "total_semantic_hits": 0,
            "total_bytes_saved": 0,
            "total_tokens_saved": 0,
            "total_cost_saved_usd": 0.0,
            "avg_latency_ms": 0,
            "cache_size_bytes": 0,
            "cache_entries": 0,
            "semantic_cache_entries": 0,
            "last_warm_time": None
        }
        
        await self.cache_manager.reset_stats()
        await self.redis_cache.reset_stats()
        await self.semantic_cache.reset_stats()
        
        logger.info("cache_service_stats_reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all cache backends."""
        return {
            "cache_manager": await self.cache_manager.ping(),
            "redis": await self.redis_cache.ping(),
            "semantic": self.semantic_cache.index.size() > 0 if self.semantic_cache.index else False,
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_cache_service = None


def get_cache_service() -> CacheService:
    """Get singleton cache service instance."""
    global _cache_service
    if not _cache_service:
        _cache_service = CacheService()
    return _cache_service


# ============================================================================
# CACHE DECORATORS
# ============================================================================

def cached(
    ttl: Optional[int] = None,
    tags: Optional[List[str]] = None,
    key_func: Optional[Callable] = None,
    use_semantic: bool = False,
    semantic_threshold: Optional[float] = None
):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        tags: Cache tags
        key_func: Function to generate cache key
        use_semantic: Whether to use semantic cache
        semantic_threshold: Similarity threshold
    
    Example:
        @cached(ttl=3600, tags=["user_profile"])
        async def get_user(user_id: str):
            return await db.fetch_user(user_id)
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_cache_service()
            
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default key: function_name:arg1:arg2:kwarg1=value1
                key_parts = [func.__name__]
                key_parts.extend([str(a) for a in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Get from cache or compute
            value, was_cached = await cache.get_or_compute(
                key=key,
                compute_func=func,
                ttl=ttl,
                tags=tags,
                use_semantic=use_semantic,
                semantic_threshold=semantic_threshold,
                *args,
                **kwargs
            )
            
            return value
        
        return wrapper
    
    return decorator


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "CacheService",
    "CacheStrategy",
    "RedisCacheStrategy",
    "SemanticCacheStrategy",
    "get_cache_service",
    "cached"
]