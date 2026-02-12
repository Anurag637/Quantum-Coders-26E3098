"""
Cache Manager - Production Ready
Centralized cache management with Redis backend, multi-level caching,
TTL management, serialization, compression, and monitoring.
"""

import json
import pickle
import zlib
import hashlib
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import Redis

from core.logging import get_logger
from core.exceptions import CacheError, CacheConnectionError
from config import settings

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# SERIALIZATION STRATEGIES
# ============================================================================

class SerializationStrategy:
    """Base class for serialization strategies."""
    
    def serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        raise NotImplementedError
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        raise NotImplementedError


class JSONSerializer(SerializationStrategy):
    """JSON serialization (human-readable, cross-platform)."""
    
    def serialize(self, value: Any) -> bytes:
        return json.dumps(value, default=str).encode('utf-8')
    
    def deserialize(self, data: bytes) -> Any:
        return json.loads(data.decode('utf-8'))


class PickleSerializer(SerializationStrategy):
    """Pickle serialization (Python-only, more efficient)."""
    
    def serialize(self, value: Any) -> bytes:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    
    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


class MessagePackSerializer(SerializationStrategy):
    """MessagePack serialization (compact, cross-platform)."""
    
    def __init__(self):
        try:
            import msgpack
            self.msgpack = msgpack
        except ImportError:
            logger.warning("msgpack not installed, falling back to JSON")
            self.fallback = JSONSerializer()
            self.use_fallback = True
        else:
            self.use_fallback = False
    
    def serialize(self, value: Any) -> bytes:
        if self.use_fallback:
            return self.fallback.serialize(value)
        return self.msgpack.packb(value, use_bin_type=True)
    
    def deserialize(self, data: bytes) -> Any:
        if self.use_fallback:
            return self.fallback.deserialize(data)
        return self.msgpack.unpackb(data, raw=False)


# ============================================================================
# COMPRESSION STRATEGIES
# ============================================================================

class CompressionStrategy:
    """Base class for compression strategies."""
    
    def compress(self, data: bytes) -> bytes:
        """Compress bytes."""
        raise NotImplementedError
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress bytes."""
        raise NotImplementedError


class ZlibCompressor(CompressionStrategy):
    """zlib compression (good balance)."""
    
    def __init__(self, level: int = 6):
        self.level = level
    
    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data, level=self.level)
    
    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)


class NoCompression(CompressionStrategy):
    """No compression."""
    
    def compress(self, data: bytes) -> bytes:
        return data
    
    def decompress(self, data: bytes) -> bytes:
        return data


# ============================================================================
# CACHE ENTRY
# ============================================================================

class CacheEntry:
    """Cache entry with metadata."""
    
    def __init__(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        version: int = 1
    ):
        self.key = key
        self.value = value
        self.created_at = datetime.utcnow()
        self.expires_at = self.created_at + timedelta(seconds=ttl) if ttl else None
        self.tags = tags or []
        self.version = version
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def record_access(self):
        """Record a cache access."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "tags": self.tags,
            "version": self.version,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        entry = cls(
            key=data["key"],
            value=data["value"],
            ttl=None,
            tags=data["tags"],
            version=data["version"]
        )
        entry.created_at = datetime.fromisoformat(data["created_at"])
        if data["expires_at"]:
            entry.expires_at = datetime.fromisoformat(data["expires_at"])
        entry.access_count = data.get("access_count", 0)
        if data.get("last_accessed"):
            entry.last_accessed = datetime.fromisoformat(data["last_accessed"])
        return entry


# ============================================================================
# CACHE STATISTICS
# ============================================================================

class CacheStats:
    """Cache statistics collector."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0
        self.total_latency_ms = 0
        self.request_count = 0
        
        # Per-key statistics
        self.key_stats = {}
    
    def record_hit(self, key: str, latency_ms: float):
        """Record a cache hit."""
        self.hits += 1
        self.request_count += 1
        self.total_latency_ms += latency_ms
        
        if key not in self.key_stats:
            self.key_stats[key] = {"hits": 0, "misses": 0, "avg_latency": 0}
        
        self.key_stats[key]["hits"] += 1
        self._update_key_latency(key, latency_ms)
    
    def record_miss(self, key: str, latency_ms: float):
        """Record a cache miss."""
        self.misses += 1
        self.request_count += 1
        self.total_latency_ms += latency_ms
        
        if key not in self.key_stats:
            self.key_stats[key] = {"hits": 0, "misses": 0, "avg_latency": 0}
        
        self.key_stats[key]["misses"] += 1
        self._update_key_latency(key, latency_ms)
    
    def record_set(self, key: str):
        """Record a cache set operation."""
        self.sets += 1
    
    def record_delete(self, key: str):
        """Record a cache delete operation."""
        self.deletes += 1
    
    def record_eviction(self, key: str):
        """Record a cache eviction."""
        self.evictions += 1
    
    def record_error(self, error: str):
        """Record a cache error."""
        self.errors += 1
    
    def _update_key_latency(self, key: str, latency_ms: float):
        """Update average latency for a key."""
        stats = self.key_stats[key]
        total_requests = stats["hits"] + stats["misses"]
        stats["avg_latency"] = (
            (stats["avg_latency"] * (total_requests - 1) + latency_ms) / total_requests
        )
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
    
    def get_avg_latency(self) -> float:
        """Get average latency."""
        return self.total_latency_ms / self.request_count if self.request_count > 0 else 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.get_hit_rate(),
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "errors": self.errors,
            "avg_latency_ms": round(self.get_avg_latency(), 2),
            "request_count": self.request_count,
            "key_count": len(self.key_stats),
            "top_keys": sorted(
                [
                    {"key": k, "hits": v["hits"], "misses": v["misses"], "avg_latency": round(v["avg_latency"], 2)}
                    for k, v in self.key_stats.items()
                ],
                key=lambda x: x["hits"],
                reverse=True
            )[:10]
        }
    
    def reset(self):
        """Reset statistics."""
        self.__init__()


# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """
    Centralized cache manager with Redis backend.
    
    Features:
    - Multi-level caching (L1: memory, L2: Redis)
    - Multiple serialization formats (JSON, Pickle, MessagePack)
    - Compression support (zlib)
    - TTL management
    - Tag-based invalidation
    - Statistics and monitoring
    - Connection pooling
    - Graceful degradation
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        max_retries: int = 3,
        retry_delay: int = 1,
        serializer: str = "json",
        compression: bool = True,
        compression_level: int = 6,
        enable_l1_cache: bool = True,
        l1_cache_size: int = 1000,
        l1_cache_ttl: int = 60,
        key_prefix: str = "llm:cache:"
    ):
        self.redis_url = redis_url or settings.redis_url
        self.default_ttl = default_ttl
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.key_prefix = key_prefix
        
        # Initialize Redis client
        self.redis_client: Optional[Redis] = None
        self._connected = False
        
        # Serialization
        self.serializers = {
            "json": JSONSerializer(),
            "pickle": PickleSerializer(),
            "msgpack": MessagePackSerializer()
        }
        self.serializer = self.serializers.get(serializer, JSONSerializer())
        
        # Compression
        if compression:
            self.compressor = ZlibCompressor(level=compression_level)
        else:
            self.compressor = NoCompression()
        
        # L1 cache (in-memory)
        self.enable_l1_cache = enable_l1_cache
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l1_cache_size = l1_cache_size
        self.l1_cache_ttl = l1_cache_ttl
        
        # Statistics
        self.stats = CacheStats()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(
            "cache_manager_initialized",
            redis_url=self.redis_url,
            default_ttl=self.default_ttl,
            serializer=serializer,
            compression=compression,
            l1_cache_enabled=self.enable_l1_cache
        )
    
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            self._connected = True
            
            # Start background cleanup
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("cache_manager_connected", redis_url=self.redis_url)
            
        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to Redis: {e}")
            raise CacheConnectionError(
                host=self.redis_url.split("@")[-1].split("/")[0],
                port=6379,
                detail=str(e)
            )
    
    async def close(self):
        """Close Redis connection."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False
            logger.info("cache_manager_closed")
    
    async def ensure_connection(self):
        """Ensure Redis connection is active."""
        if not self._connected or not self.redis_client:
            await self.initialize()
    
    async def ping(self) -> bool:
        """Check if Redis is responsive."""
        try:
            if self.redis_client:
                return await self.redis_client.ping()
            return False
        except Exception:
            return False
    
    # ========================================================================
    # CORE OPERATIONS
    # ========================================================================
    
    async def get(
        self,
        key: str,
        default: Any = None,
        use_l1: bool = True
    ) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            use_l1: Whether to check L1 cache
        
        Returns:
            Cached value or default
        """
        start_time = datetime.utcnow()
        full_key = self._make_key(key)
        
        try:
            # Check L1 cache first
            if use_l1 and self.enable_l1_cache and key in self.l1_cache:
                entry = self.l1_cache[key]
                
                if not entry.is_expired():
                    entry.record_access()
                    latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                    self.stats.record_hit(key, latency)
                    
                    logger.debug("l1_cache_hit", key=key)
                    return entry.value
                else:
                    # Remove expired entry
                    del self.l1_cache[key]
            
            # Check Redis
            await self.ensure_connection()
            
            data = await self.redis_client.get(full_key)
            
            if data:
                # Decompress and deserialize
                decompressed = self.compressor.decompress(data)
                entry_dict = self.serializer.deserialize(decompressed)
                entry = CacheEntry.from_dict(entry_dict)
                
                if not entry.is_expired():
                    entry.record_access()
                    
                    # Update L1 cache
                    if self.enable_l1_cache:
                        self._update_l1_cache(key, entry)
                    
                    latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                    self.stats.record_hit(key, latency)
                    
                    logger.debug("redis_cache_hit", key=key)
                    return entry.value
            
            # Cache miss
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.stats.record_miss(key, latency)
            
            return default
            
        except Exception as e:
            self.stats.record_error(str(e))
            logger.error(f"Cache get failed: {key} - {e}")
            
            # Fail open - return default
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        use_l1: bool = True
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Tags for group invalidation
            use_l1: Whether to update L1 cache
        
        Returns:
            True if successful
        """
        full_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        
        try:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                tags=tags or []
            )
            
            # Serialize and compress
            entry_dict = entry.to_dict()
            serialized = self.serializer.serialize(entry_dict)
            compressed = self.compressor.compress(serialized)
            
            # Store in Redis
            await self.ensure_connection()
            
            await self.redis_client.setex(
                full_key,
                ttl,
                compressed
            )
            
            # Update L1 cache
            if use_l1 and self.enable_l1_cache:
                self._update_l1_cache(key, entry, ttl=min(ttl, self.l1_cache_ttl))
            
            # Store tag mappings
            if tags:
                await self._add_tag_mappings(key, tags, ttl)
            
            self.stats.record_set(key)
            
            logger.debug("cache_set", key=key, ttl=ttl, tags=tags)
            return True
            
        except Exception as e:
            self.stats.record_error(str(e))
            logger.error(f"Cache set failed: {key} - {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if key was deleted
        """
        full_key = self._make_key(key)
        
        try:
            # Remove from L1 cache
            if key in self.l1_cache:
                del self.l1_cache[key]
            
            # Remove from Redis
            await self.ensure_connection()
            
            result = await self.redis_client.delete(full_key)
            
            if result > 0:
                self.stats.record_delete(key)
                logger.debug("cache_delete", key=key)
                return True
            
            return False
            
        except Exception as e:
            self.stats.record_error(str(e))
            logger.error(f"Cache delete failed: {key} - {e}")
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries matching pattern.
        
        Args:
            pattern: Key pattern to clear (e.g., "user:*")
        
        Returns:
            Number of keys cleared
        """
        try:
            await self.ensure_connection()
            
            if pattern:
                full_pattern = self._make_key(pattern)
                keys = await self.redis_client.keys(full_pattern)
            else:
                # Clear all cache keys
                keys = await self.redis_client.keys(f"{self.key_prefix}*")
            
            if keys:
                # Clear from Redis
                result = await self.redis_client.delete(*keys)
                
                # Clear from L1 cache
                for key in keys:
                    clean_key = key.decode('utf-8').replace(self.key_prefix, '', 1)
                    if clean_key in self.l1_cache:
                        del self.l1_cache[clean_key]
                
                logger.info("cache_cleared", keys_removed=result, pattern=pattern)
                return result
            
            return 0
            
        except Exception as e:
            self.stats.record_error(str(e))
            logger.error(f"Cache clear failed: {e}")
            return 0
    
    # ========================================================================
    # BATCH OPERATIONS
    # ========================================================================
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple keys from cache.
        
        Args:
            keys: List of cache keys
        
        Returns:
            Dictionary of key-value pairs
        """
        results = {}
        
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        
        return results
    
    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Set multiple key-value pairs.
        
        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time to live in seconds
            tags: Tags for group invalidation
        
        Returns:
            True if all sets successful
        """
        success = True
        
        for key, value in mapping.items():
            if not await self.set(key, value, ttl, tags):
                success = False
        
        return success
    
    async def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple keys.
        
        Args:
            keys: List of cache keys
        
        Returns:
            Number of keys deleted
        """
        deleted = 0
        
        for key in keys:
            if await self.delete(key):
                deleted += 1
        
        return deleted
    
    # ========================================================================
    # TAG-BASED OPERATIONS
    # ========================================================================
    
    async def get_by_tag(self, tag: str) -> List[Any]:
        """
        Get all values with a specific tag.
        
        Args:
            tag: Tag to search for
        
        Returns:
            List of cached values
        """
        try:
            tag_key = f"{self.key_prefix}tag:{tag}"
            keys = await self.redis_client.smembers(tag_key)
            
            values = []
            for key in keys:
                key_str = key.decode('utf-8').replace(self.key_prefix, '', 1)
                value = await self.get(key_str)
                if value is not None:
                    values.append(value)
            
            return values
            
        except Exception as e:
            self.stats.record_error(str(e))
            logger.error(f"Get by tag failed: {tag} - {e}")
            return []
    
    async def delete_by_tag(self, tag: str) -> int:
        """
        Delete all keys with a specific tag.
        
        Args:
            tag: Tag to delete
        
        Returns:
            Number of keys deleted
        """
        try:
            tag_key = f"{self.key_prefix}tag:{tag}"
            keys = await self.redis_client.smembers(tag_key)
            
            deleted = 0
            for key in keys:
                key_str = key.decode('utf-8').replace(self.key_prefix, '', 1)
                if await self.delete(key_str):
                    deleted += 1
            
            # Delete the tag set
            await self.redis_client.delete(tag_key)
            
            logger.info("cache_deleted_by_tag", tag=tag, keys_deleted=deleted)
            return deleted
            
        except Exception as e:
            self.stats.record_error(str(e))
            logger.error(f"Delete by tag failed: {tag} - {e}")
            return 0
    
    async def _add_tag_mappings(
        self,
        key: str,
        tags: List[str],
        ttl: int
    ):
        """Add tag mappings for a key."""
        try:
            for tag in tags:
                tag_key = f"{self.key_prefix}tag:{tag}"
                full_key = self._make_key(key)
                
                # Add key to tag set
                await self.redis_client.sadd(tag_key, full_key)
                
                # Set TTL on tag set
                await self.redis_client.expire(tag_key, ttl)
                
        except Exception as e:
            logger.error(f"Failed to add tag mappings: {e}")
    
    # ========================================================================
    # L1 CACHE MANAGEMENT
    # ========================================================================
    
    def _update_l1_cache(self, key: str, entry: CacheEntry, ttl: Optional[int] = None):
        """Update L1 cache with size management."""
        # Enforce size limit
        if len(self.l1_cache) >= self.l1_cache_size:
            # Remove oldest entry (simple LRU approximation)
            oldest_key = min(
                self.l1_cache.keys(),
                key=lambda k: self.l1_cache[k].last_accessed
            )
            del self.l1_cache[oldest_key]
            self.stats.record_eviction(oldest_key)
        
        # Adjust TTL for L1 cache
        if ttl:
            entry.expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        
        self.l1_cache[key] = entry
    
    async def _cleanup_loop(self):
        """Background task to clean up expired L1 cache entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                expired_keys = []
                for key, entry in self.l1_cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.l1_cache[key]
                    logger.debug("l1_cache_expired", key=key)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix."""
        return f"{self.key_prefix}{key}"
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            # Check L1 cache
            if key in self.l1_cache and not self.l1_cache[key].is_expired():
                return True
            
            # Check Redis
            await self.ensure_connection()
            full_key = self._make_key(key)
            return await self.redis_client.exists(full_key) > 0
            
        except Exception as e:
            self.stats.record_error(str(e))
            logger.error(f"Cache exists failed: {key} - {e}")
            return False
    
    async def ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key in seconds."""
        try:
            await self.ensure_connection()
            full_key = self._make_key(key)
            return await self.redis_client.ttl(full_key)
        except Exception as e:
            self.stats.record_error(str(e))
            logger.error(f"Cache ttl failed: {key} - {e}")
            return None
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on existing key."""
        try:
            await self.ensure_connection()
            full_key = self._make_key(key)
            return await self.redis_client.expire(full_key, ttl)
        except Exception as e:
            self.stats.record_error(str(e))
            logger.error(f"Cache expire failed: {key} - {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern."""
        try:
            await self.ensure_connection()
            full_pattern = self._make_key(pattern)
            keys = await self.redis_client.keys(full_pattern)
            
            # Remove prefix
            return [
                k.decode('utf-8').replace(self.key_prefix, '', 1)
                for k in keys
            ]
        except Exception as e:
            self.stats.record_error(str(e))
            logger.error(f"Cache keys failed: {pattern} - {e}")
            return []
    
    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter."""
        try:
            await self.ensure_connection()
            full_key = self._make_key(key)
            return await self.redis_client.incrby(full_key, amount)
        except Exception as e:
            self.stats.record_error(str(e))
            logger.error(f"Cache incr failed: {key} - {e}")
            return None
    
    async def decr(self, key: str, amount: int = 1) -> Optional[int]:
        """Decrement counter."""
        try:
            await self.ensure_connection()
            full_key = self._make_key(key)
            return await self.redis_client.decrby(full_key, amount)
        except Exception as e:
            self.stats.record_error(str(e))
            logger.error(f"Cache decr failed: {key} - {e}")
            return None
    
    # ========================================================================
    # STATISTICS AND MONITORING
    # ========================================================================
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.get_stats()
        
        # Add Redis info
        try:
            if self.redis_client:
                info = await self.redis_client.info()
                stats["redis"] = {
                    "version": info.get("redis_version", "unknown"),
                    "used_memory_human": info.get("used_memory_human", "0"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "uptime_in_seconds": info.get("uptime_in_seconds", 0)
                }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
        
        # Add L1 cache info
        stats["l1_cache"] = {
            "size": len(self.l1_cache),
            "max_size": self.l1_cache_size,
            "usage_percent": (len(self.l1_cache) / self.l1_cache_size * 100) if self.l1_cache_size > 0 else 0,
            "enabled": self.enable_l1_cache,
            "ttl": self.l1_cache_ttl
        }
        
        stats["config"] = {
            "default_ttl": self.default_ttl,
            "serializer": self.serializer.__class__.__name__,
            "compression": self.compressor.__class__.__name__,
            "connected": self._connected
        }
        
        return stats
    
    async def reset_stats(self):
        """Reset cache statistics."""
        self.stats.reset()
        logger.info("cache_stats_reset")
    
    def get_size(self) -> str:
        """Get approximate cache size (placeholder)."""
        # This would require scanning all keys
        return "N/A"


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get singleton cache manager instance."""
    global _cache_manager
    if not _cache_manager:
        _cache_manager = CacheManager()
    return _cache_manager


# ============================================================================
# CACHE DECORATOR
# ============================================================================

def cached(
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
    tags: Optional[List[str]] = None
):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_func: Function to generate cache key from args/kwargs
        tags: Tags for group invalidation
    
    Example:
        @cached(ttl=3600, tags=["user_profile"])
        async def get_user(user_id: str):
            return await db.fetch_user(user_id)
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default key: function_name:arg1:arg2:kwarg1=value1
                key_parts = [func.__name__]
                key_parts.extend([str(a) for a in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try cache
            cached_value = await cache.get(key)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(key, result, ttl=ttl, tags=tags)
            
            return result
        
        return wrapper
    
    return decorator


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "CacheManager",
    "CacheEntry",
    "CacheStats",
    "get_cache_manager",
    "cached",
    
    # Serialization
    "SerializationStrategy",
    "JSONSerializer",
    "PickleSerializer",
    "MessagePackSerializer",
    
    # Compression
    "CompressionStrategy",
    "ZlibCompressor",
    "NoCompression"
]