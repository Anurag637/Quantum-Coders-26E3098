"""
Redis Cache Implementation - Production Ready
High-performance Redis client with connection pooling, retry logic,
health checks, and comprehensive monitoring.
"""

import json
import time
import asyncio
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError, ConnectionError, TimeoutError

from core.logging import get_logger
from core.exceptions import CacheConnectionError, CacheError
from config import settings

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# REDIS CONNECTION POOL
# ============================================================================

class RedisConnectionPool:
    """
    Manages Redis connection pool with health checks and automatic recovery.
    
    Features:
    - Connection pooling for efficiency
    - Health checks to detect dead connections
    - Automatic reconnection
    - Pool statistics
    """
    
    def __init__(
        self,
        url: str,
        max_connections: int = 50,
        timeout: int = 5,
        socket_connect_timeout: int = 5,
        socket_keepalive: bool = True,
        health_check_interval: int = 30,
        retry_on_timeout: bool = True,
        decode_responses: bool = False
    ):
        self.url = url
        self.max_connections = max_connections
        self.timeout = timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.socket_keepalive = socket_keepalive
        self.health_check_interval = health_check_interval
        self.retry_on_timeout = retry_on_timeout
        self.decode_responses = decode_responses
        
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = None
        self._connected = False
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "created_connections": 0,
            "acquired_connections": 0,
            "released_connections": 0,
            "failed_connections": 0,
            "health_check_failures": 0,
            "reconnections": 0
        }
    
    async def initialize(self) -> Redis:
        """Initialize connection pool and client."""
        async with self._lock:
            try:
                # Create connection pool
                self._pool = ConnectionPool.from_url(
                    url=self.url,
                    max_connections=self.max_connections,
                    timeout=self.timeout,
                    socket_connect_timeout=self.socket_connect_timeout,
                    socket_keepalive=self.socket_keepalive,
                    health_check_interval=self.health_check_interval,
                    retry_on_timeout=self.retry_on_timeout,
                    decode_responses=self.decode_responses
                )
                
                # Create client
                self._client = Redis(connection_pool=self._pool)
                
                # Test connection
                await self._client.ping()
                self._connected = True
                
                self.stats["created_connections"] += 1
                
                logger.info(
                    "redis_pool_initialized",
                    url=self.url,
                    max_connections=self.max_connections
                )
                
                return self._client
                
            except Exception as e:
                self.stats["failed_connections"] += 1
                logger.error(f"Failed to initialize Redis pool: {e}")
                raise CacheConnectionError(
                    host=self.url.split("@")[-1].split("/")[0],
                    port=6379,
                    detail=str(e)
                )
    
    async def get_client(self) -> Redis:
        """Get Redis client, reconnecting if necessary."""
        if not self._connected or not self._client:
            return await self.initialize()
        
        try:
            # Test connection
            await self._client.ping()
        except Exception:
            # Reconnect on failure
            logger.warning("Redis connection lost, reconnecting...")
            self.stats["reconnections"] += 1
            return await self.initialize()
        
        return self._client
    
    async def close(self):
        """Close connection pool."""
        async with self._lock:
            if self._pool:
                await self._pool.aclose()
                self._pool = None
                self._client = None
                self._connected = False
                logger.info("Redis connection pool closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        stats = self.stats.copy()
        
        if self._pool:
            pool_stats = {
                "max_connections": self._pool.max_connections,
                "in_use_connections": self._pool._in_use_connections,
                "available_connections": self._pool._available_connections,
                "created_connections": self._pool._created_connections
            }
            stats.update(pool_stats)
        
        return stats


# ============================================================================
# REDIS CACHE
# ============================================================================

class RedisCache:
    """
    High-performance Redis cache implementation.
    
    Features:
    - Connection pooling
    - Automatic retry with backoff
    - Health checks
    - Comprehensive monitoring
    - Batch operations
    - Pub/Sub support
    - LUA scripting
    - Cluster support (optional)
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        max_connections: int = 50,
        default_ttl: int = 3600,
        retry_attempts: int = 3,
        retry_delay: float = 0.1,
        retry_backoff: float = 2.0,
        enable_health_checks: bool = True,
        health_check_interval: int = 30,
        key_prefix: str = "llm:cache:"
    ):
        self.url = url or settings.redis_url
        self.default_ttl = default_ttl
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.enable_health_checks = enable_health_checks
        self.key_prefix = key_prefix
        
        # Connection pool
        self.pool = RedisConnectionPool(
            url=self.url,
            max_connections=max_connections,
            health_check_interval=health_check_interval
        )
        
        # Client (lazy initialized)
        self._client: Optional[Redis] = None
        
        # Health check task
        self._health_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "retry_count": 0,
            "total_latency_ms": 0
        }
        
        logger.info(
            "redis_cache_initialized",
            url=self.url,
            max_connections=max_connections,
            default_ttl=self.default_ttl
        )
    
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    async def initialize(self):
        """Initialize Redis connection."""
        self._client = await self.pool.initialize()
        
        # Start health checks
        if self.enable_health_checks:
            self._health_task = asyncio.create_task(self._health_check_loop())
    
    async def close(self):
        """Close Redis connection."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        await self.pool.close()
        self._client = None
    
    async def _get_client(self) -> Redis:
        """Get Redis client with lazy initialization."""
        if not self._client:
            await self.initialize()
        return await self.pool.get_client()
    
    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if self._client:
                    await self._client.ping()
                    logger.debug("Redis health check passed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.pool.stats["health_check_failures"] += 1
                logger.error(f"Redis health check failed: {e}")
                
                # Attempt reconnection
                try:
                    await self.initialize()
                    logger.info("Redis reconnected successfully")
                except Exception as reconnect_error:
                    logger.error(f"Redis reconnection failed: {reconnect_error}")
    
    # ========================================================================
    # RETRY LOGIC
    # ========================================================================
    
    async def _execute_with_retry(
        self,
        operation: str,
        func: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute Redis operation with retry logic.
        
        Args:
            operation: Operation name for logging
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            CacheError: If all retry attempts fail
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                client = await self._get_client()
                result = await func(client, *args, **kwargs)
                
                # Record success
                self.stats["total_operations"] += 1
                self.stats["successful_operations"] += 1
                self.stats["total_latency_ms"] += (time.time() - start_time) * 1000
                
                return result
                
            except (ConnectionError, TimeoutError) as e:
                last_error = e
                self.stats["retry_count"] += 1
                
                # Calculate backoff delay
                delay = self.retry_delay * (self.retry_backoff ** attempt)
                
                logger.warning(
                    f"Redis {operation} failed (attempt {attempt + 1}/{self.retry_attempts}): {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                await asyncio.sleep(delay)
                
            except RedisError as e:
                # Non-retryable Redis error
                self.stats["failed_operations"] += 1
                self.stats["total_operations"] += 1
                self.stats["total_latency_ms"] += (time.time() - start_time) * 1000
                
                logger.error(f"Redis {operation} failed: {e}")
                raise CacheError(f"Redis {operation} failed: {e}")
        
        # All retry attempts failed
        self.stats["failed_operations"] += 1
        self.stats["total_operations"] += 1
        self.stats["total_latency_ms"] += (time.time() - start_time) * 1000
        
        raise CacheError(
            f"Redis {operation} failed after {self.retry_attempts} attempts: {last_error}"
        )
    
    # ========================================================================
    # KEY OPERATIONS
    # ========================================================================
    
    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[bytes]:
        """Get value by key."""
        full_key = self._make_key(key)
        
        async def _get(client: Redis):
            return await client.get(full_key)
        
        return await self._execute_with_retry("GET", _get)
    
    async def set(
        self,
        key: str,
        value: Union[str, bytes],
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set key to value with optional TTL."""
        full_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        
        async def _set(client: Redis):
            if nx:
                return await client.set(full_key, value, nx=True, ex=ttl)
            elif xx:
                return await client.set(full_key, value, xx=True, ex=ttl)
            else:
                return await client.setex(full_key, ttl, value)
        
        result = await self._execute_with_retry("SET", _set)
        return bool(result)
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        full_keys = [self._make_key(key) for key in keys]
        
        async def _delete(client: Redis):
            return await client.delete(*full_keys)
        
        return await self._execute_with_retry("DELETE", _delete)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        full_key = self._make_key(key)
        
        async def _exists(client: Redis):
            return await client.exists(full_key)
        
        result = await self._execute_with_retry("EXISTS", _exists)
        return result > 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on key."""
        full_key = self._make_key(key)
        
        async def _expire(client: Redis):
            return await client.expire(full_key, ttl)
        
        return await self._execute_with_retry("EXPIRE", _expire)
    
    async def ttl(self, key: str) -> int:
        """Get remaining TTL for key."""
        full_key = self._make_key(key)
        
        async def _ttl(client: Redis):
            return await client.ttl(full_key)
        
        return await self._execute_with_retry("TTL", _ttl)
    
    async def persist(self, key: str) -> bool:
        """Remove expiration from key."""
        full_key = self._make_key(key)
        
        async def _persist(client: Redis):
            return await client.persist(full_key)
        
        return await self._execute_with_retry("PERSIST", _persist)
    
    # ========================================================================
    # BATCH OPERATIONS
    # ========================================================================
    
    async def mget(self, keys: List[str]) -> List[Optional[bytes]]:
        """Get multiple keys."""
        full_keys = [self._make_key(key) for key in keys]
        
        async def _mget(client: Redis):
            return await client.mget(*full_keys)
        
        return await self._execute_with_retry("MGET", _mget)
    
    async def mset(self, mapping: Dict[str, Union[str, bytes]], ttl: Optional[int] = None) -> bool:
        """Set multiple keys."""
        full_mapping = {
            self._make_key(key): value
            for key, value in mapping.items()
        }
        
        async def _mset(client: Redis):
            result = await client.mset(full_mapping)
            
            # Set TTL for each key if specified
            if ttl:
                for key in full_mapping.keys():
                    await client.expire(key, ttl)
            
            return result
        
        return await self._execute_with_retry("MSET", _mset)
    
    # ========================================================================
    # HASH OPERATIONS
    # ========================================================================
    
    async def hget(self, key: str, field: str) -> Optional[bytes]:
        """Get hash field."""
        full_key = self._make_key(key)
        
        async def _hget(client: Redis):
            return await client.hget(full_key, field)
        
        return await self._execute_with_retry("HGET", _hget)
    
    async def hset(self, key: str, field: str, value: Union[str, bytes]) -> int:
        """Set hash field."""
        full_key = self._make_key(key)
        
        async def _hset(client: Redis):
            return await client.hset(full_key, field, value)
        
        return await self._execute_with_retry("HSET", _hset)
    
    async def hgetall(self, key: str) -> Dict[str, bytes]:
        """Get all hash fields."""
        full_key = self._make_key(key)
        
        async def _hgetall(client: Redis):
            return await client.hgetall(full_key)
        
        result = await self._execute_with_retry("HGETALL", _hgetall)
        
        # Decode bytes keys to strings
        return {
            k.decode('utf-8') if isinstance(k, bytes) else k: v
            for k, v in result.items()
        }
    
    async def hdel(self, key: str, *fields: str) -> int:
        """Delete hash fields."""
        full_key = self._make_key(key)
        
        async def _hdel(client: Redis):
            return await client.hdel(full_key, *fields)
        
        return await self._execute_with_retry("HDEL", _hdel)
    
    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        """Increment hash field."""
        full_key = self._make_key(key)
        
        async def _hincrby(client: Redis):
            return await client.hincrby(full_key, field, amount)
        
        return await self._execute_with_retry("HINCRBY", _hincrby)
    
    # ========================================================================
    # LIST OPERATIONS
    # ========================================================================
    
    async def lpush(self, key: str, *values: Union[str, bytes]) -> int:
        """Prepend values to list."""
        full_key = self._make_key(key)
        
        async def _lpush(client: Redis):
            return await client.lpush(full_key, *values)
        
        return await self._execute_with_retry("LPUSH", _lpush)
    
    async def rpush(self, key: str, *values: Union[str, bytes]) -> int:
        """Append values to list."""
        full_key = self._make_key(key)
        
        async def _rpush(client: Redis):
            return await client.rpush(full_key, *values)
        
        return await self._execute_with- -retry("RPUSH", _rpush)
    
    async def lpop(self, key: str) -> Optional[bytes]:
        """Remove and get first element."""
        full_key = self._make_key(key)
        
        async def _lpop(client: Redis):
            return await client.lpop(full_key)
        
        return await self._execute_with_retry("LPOP", _lpop)
    
    async def rpop(self, key: str) -> Optional[bytes]:
        """Remove and get last element."""
        full_key = self._make_key(key)
        
        async def _rpop(client: Redis):
            return await client.rpop(full_key)
        
        return await self._execute_with_retry("RPOP", _rpop)
    
    async def lrange(self, key: str, start: int, stop: int) -> List[bytes]:
        """Get range of elements."""
        full_key = self._make_key(key)
        
        async def _lrange(client: Redis):
            return await client.lrange(full_key, start, stop)
        
        return await self._execute_with_retry("LRANGE", _lrange)
    
    async def llen(self, key: str) -> int:
        """Get list length."""
        full_key = self._make_key(key)
        
        async def _llen(client: Redis):
            return await client.llen(full_key)
        
        return await self._execute_with_retry("LLEN", _llen)
    
    # ========================================================================
    # SET OPERATIONS
    # ========================================================================
    
    async def sadd(self, key: str, *members: Union[str, bytes]) -> int:
        """Add members to set."""
        full_key = self._make_key(key)
        
        async def _sadd(client: Redis):
            return await client.sadd(full_key, *members)
        
        return await self._execute_with_retry("SADD", _sadd)
    
    async def srem(self, key: str, *members: Union[str, bytes]) -> int:
        """Remove members from set."""
        full_key = self._make_key(key)
        
        async def _srem(client: Redis):
            return await client.srem(full_key, *members)
        
        return await self._execute_with_retry("SREM", _srem)
    
    async def smembers(self, key: str) -> List[bytes]:
        """Get all set members."""
        full_key = self._make_key(key)
        
        async def _smembers(client: Redis):
            return await client.smembers(full_key)
        
        result = await self._execute_with_retry("SMEMBERS", _smembers)
        return list(result)
    
    async def sismember(self, key: str, member: Union[str, bytes]) -> bool:
        """Check if member exists in set."""
        full_key = self._make_key(key)
        
        async def _sismember(client: Redis):
            return await client.sismember(full_key, member)
        
        return await self._execute_with_retry("SISMEMBER", _sismember)
    
    async def scard(self, key: str) -> int:
        """Get set cardinality."""
        full_key = self._make_key(key)
        
        async def _scard(client: Redis):
            return await client.scard(full_key)
        
        return await self._execute_with_retry("SCARD", _scard)
    
    # ========================================================================
    # SORTED SET OPERATIONS
    # ========================================================================
    
    async def zadd(self, key: str, mapping: Dict[Union[str, bytes], float]) -> int:
        """Add members to sorted set with scores."""
        full_key = self._make_key(key)
        
        async def _zadd(client: Redis):
            return await client.zadd(full_key, mapping)
        
        return await self._execute_with_retry("ZADD", _zadd)
    
    async def zrange(
        self,
        key: str,
        start: int,
        stop: int,
        withscores: bool = False,
        desc: bool = False
    ) -> Union[List[bytes], List[Tuple[bytes, float]]]:
        """Get range of members by index."""
        full_key = self._make_key(key)
        
        async def _zrange(client: Redis):
            return await client.zrange(
                full_key,
                start,
                stop,
                withscores=withscores,
                desc=desc
            )
        
        return await self._execute_with_retry("ZRANGE", _zrange)
    
    async def zrangebyscore(
        self,
        key: str,
        min: Union[float, str],
        max: Union[float, str],
        withscores: bool = False,
        offset: int = 0,
        count: Optional[int] = None
    ) -> Union[List[bytes], List[Tuple[bytes, float]]]:
        """Get range of members by score."""
        full_key = self._make_key(key)
        
        async def _zrangebyscore(client: Redis):
            return await client.zrangebyscore(
                full_key,
                min,
                max,
                withscores=withscores,
                offset=offset,
                count=count
            )
        
        return await self._execute_with_retry("ZRANGEBYSCORE", _zrangebyscore)
    
    async def zrem(self, key: str, *members: Union[str, bytes]) -> int:
        """Remove members from sorted set."""
        full_key = self._make_key(key)
        
        async def _zrem(client: Redis):
            return await client.zrem(full_key, *members)
        
        return await self._execute_with_retry("ZREM", _zrem)
    
    async def zcard(self, key: str) -> int:
        """Get sorted set cardinality."""
        full_key = self._make_key(key)
        
        async def _zcard(client: Redis):
            return await client.zcard(full_key)
        
        return await self._execute_with_retry("ZCARD", _zcard)
    
    async def zscore(self, key: str, member: Union[str, bytes]) -> Optional[float]:
        """Get score of member."""
        full_key = self._make_key(key)
        
        async def _zscore(client: Redis):
            return await client.zscore(full_key, member)
        
        return await self._execute_with_retry("ZSCORE", _zscore)
    
    # ========================================================================
    # PUB/SUB OPERATIONS
    # ========================================================================
    
    async def publish(self, channel: str, message: Union[str, bytes]) -> int:
        """Publish message to channel."""
        async def _publish(client: Redis):
            return await client.publish(channel, message)
        
        return await self._execute_with_retry("PUBLISH", _publish)
    
    @asynccontextmanager
    async def subscribe(self, channel: str):
        """Subscribe to channel (context manager)."""
        client = await self._get_client()
        pubsub = client.pubsub()
        
        try:
            await pubsub.subscribe(channel)
            yield pubsub
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
    
    # ========================================================================
    # LUA SCRIPTING
    # ========================================================================
    
    async def eval(
        self,
        script: str,
        keys: List[str] = None,
        args: List[str] = None
    ) -> Any:
        """Execute LUA script."""
        keys = keys or []
        args = args or []
        
        # Add prefix to keys
        full_keys = [self._make_key(key) for key in keys]
        
        async def _eval(client: Redis):
            return await client.eval(script, len(full_keys), *(full_keys + args))
        
        return await self._execute_with_retry("EVAL", _eval)
    
    async def evalsha(
        self,
        sha: str,
        keys: List[str] = None,
        args: List[str] = None
    ) -> Any:
        """Execute LUA script by SHA."""
        keys = keys or []
        args = args or []
        
        # Add prefix to keys
        full_keys = [self._make_key(key) for key in keys]
        
        async def _evalsha(client: Redis):
            return await client.evalsha(sha, len(full_keys), *(full_keys + args))
        
        return await self._execute_with_retry("EVALSHA", _evalsha)
    
    async def script_load(self, script: str) -> str:
        """Load LUA script into Redis."""
        async def _script_load(client: Redis):
            return await client.script_load(script)
        
        return await self._execute_with- -retry("SCRIPT_LOAD", _script_load)
    
    # ========================================================================
    # SERVER INFORMATION
    # ========================================================================
    
    async def info(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get Redis server information."""
        async def _info(client: Redis):
            return await client.info(section)
        
        return await self._execute_with_retry("INFO", _info)
    
    async def dbsize(self) -> int:
        """Get number of keys in database."""
        async def _dbsize(client: Redis):
            return await client.dbsize()
        
        return await self._execute_with_retry("DBSIZE", _dbsize)
    
    async def flushdb(self) -> bool:
        """Flush current database."""
        async def _flushdb(client: Redis):
            return await client.flushdb()
        
        return await self._execute_with_retry("FLUSHDB", _flushdb)
    
    async def flushall(self) -> bool:
        """Flush all databases."""
        async def _flushall(client: Redis):
            return await client.flushall()
        
        return await self._execute_with_retry("FLUSHALL", _flushall)
    
    # ========================================================================
    # PIPELINE OPERATIONS
    # ========================================================================
    
    @asynccontextmanager
    async def pipeline(self, transaction: bool = True):
        """Create Redis pipeline."""
        client = await self._get_client()
        pipe = client.pipeline(transaction=transaction)
        
        try:
            yield pipe
            await pipe.execute()
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
        finally:
            await pipe.reset()
    
    # ========================================================================
    # MONITORING AND STATISTICS
    # ========================================================================
    
    async def ping(self) -> bool:
        """Ping Redis server."""
        try:
            client = await self._get_client()
            return await client.ping()
        except Exception:
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "operations": self.stats.copy(),
            "pool": self.pool.get_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add server info
        try:
            info = await self.info()
            stats["server"] = {
                "version": info.get("redis_version", "unknown"),
                "mode": info.get("redis_mode", "unknown"),
                "os": info.get("os", "unknown"),
                "arch_bits": info.get("arch_bits", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0"),
                "total_keys": await self.dbsize()
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            stats["server"] = {"error": str(e)}
        
        return stats
    
    async def reset_stats(self):
        """Reset operation statistics."""
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "retry_count": 0,
            "total_latency_ms": 0
        }
        logger.info("Redis cache statistics reset")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_redis_cache = None


def get_redis_cache() -> RedisCache:
    """Get singleton Redis cache instance."""
    global _redis_cache
    if not _redis_cache:
        _redis_cache = RedisCache()
    return _redis_cache


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "RedisCache",
    "RedisConnectionPool",
    "get_redis_cache"
]