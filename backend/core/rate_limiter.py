"""
Rate Limiting System - Production Ready
Distributed rate limiting with Redis backend, multiple strategies,
and comprehensive monitoring.
"""

import time
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List, Callable, Union
from functools import wraps

import redis.asyncio as redis
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer

from config import settings
from core.logging import get_logger
from core.exceptions import RateLimitExceededError
from core.security import get_rate_limit_key, get_current_user_required

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# RATE LIMIT STRATEGIES
# ============================================================================

class RateLimitStrategy:
    """Base class for rate limiting strategies."""
    
    async def check(
        self,
        redis_client: redis.Redis,
        key: str,
        limit: int,
        window: int
    ) -> Tuple[bool, int, int]:
        """
        Check if request is within rate limit.
        
        Args:
            redis_client: Redis client
            key: Rate limit key
            limit: Maximum requests per window
            window: Time window in seconds
        
        Returns:
            Tuple of (is_allowed, remaining, reset_time)
        """
        raise NotImplementedError


class FixedWindowStrategy(RateLimitStrategy):
    """
    Fixed window rate limiting.
    
    Pros: Simple, memory efficient
    Cons: Can allow double the requests at window boundaries
    """
    
    async def check(
        self,
        redis_client: redis.Redis,
        key: str,
        limit: int,
        window: int
    ) -> Tuple[bool, int, int]:
        current = int(time.time())
        window_key = f"{key}:{current // window}"
        
        # Increment counter
        count = await redis_client.incr(window_key)
        
        # Set expiry on first request
        if count == 1:
            await redis_client.expire(window_key, window)
        
        # Calculate reset time
        reset_time = ((current // window) + 1) * window
        
        is_allowed = count <= limit
        remaining = max(0, limit - count)
        
        return is_allowed, remaining, reset_time


class SlidingWindowStrategy(RateLimitStrategy):
    """
    Sliding window rate limiting using sorted sets.
    
    Pros: Accurate, no boundary issues
    Cons: More memory, slightly slower
    """
    
    async def check(
        self,
        redis_client: redis.Redis,
        key: str,
        limit: int,
        window: int
    ) -> Tuple[bool, int, int]:
        now = time.time()
        window_start = now - window
        
        # Use sorted set to store timestamps
        async with redis_client.pipeline() as pipe:
            # Remove old entries
            await pipe.zremrangebyscore(key, 0, window_start)
            # Count current entries
            await pipe.zcard(key)
            # Add current request
            await pipe.zadd(key, {str(now): now})
            # Set expiry
            await pipe.expire(key, window * 2)
            
            _, count, _, _ = await pipe.execute()
        
        is_allowed = count <= limit
        remaining = max(0, limit - count)
        reset_time = int(now + window)
        
        return is_allowed, remaining, reset_time


class TokenBucketStrategy(RateLimitStrategy):
    """
    Token bucket rate limiting.
    
    Pros: Allows bursts, smooth consumption
    Cons: More complex, requires state
    """
    
    async def check(
        self,
        redis_client: redis.Redis,
        key: str,
        limit: int,
        window: int
    ) -> Tuple[bool, int, int]:
        now = time.time()
        bucket_key = f"{key}:bucket"
        
        # Rate: tokens per second
        rate = limit / window
        
        async with redis_client.pipeline() as pipe:
            # Get current bucket state
            await pipe.hgetall(bucket_key)
            result = await pipe.execute()
            
            if result[0]:
                tokens = float(result[0].get(b'tokens', limit))
                last_refill = float(result[0].get(b'last_refill', now))
            else:
                tokens = limit
                last_refill = now
            
            # Refill tokens based on time passed
            time_passed = now - last_refill
            new_tokens = time_passed * rate
            tokens = min(limit, tokens + new_tokens)
            
            # Check if we have enough tokens
            if tokens >= 1:
                tokens -= 1
                is_allowed = True
                remaining = int(tokens)
            else:
                is_allowed = False
                remaining = 0
            
            # Store updated bucket
            await pipe.hset(bucket_key, mapping={
                'tokens': tokens,
                'last_refill': now
            })
            await pipe.expire(bucket_key, window * 2)
            await pipe.execute()
        
        reset_time = int(now + (1 / rate))  # Time until next token
        return is_allowed, remaining, reset_time


# ============================================================================
# RATE LIMIT TIERS
# ============================================================================

class RateLimitTier:
    """Rate limit tiers for different user types."""
    
    # Anonymous users
    ANONYMOUS = {
        "requests": 20,
        "period": 60,  # 20 requests per minute
        "burst": 5
    }
    
    # Free tier
    FREE = {
        "requests": 100,
        "period": 60,  # 100 requests per minute
        "burst": 20
    }
    
    # Pro tier
    PRO = {
        "requests": 1000,
        "period": 60,  # 1000 requests per minute
        "burst": 100
    }
    
    # Enterprise tier
    ENTERPRISE = {
        "requests": 10000,
        "period": 60,  # 10000 requests per minute
        "burst": 1000
    }
    
    # Admin tier
    ADMIN = {
        "requests": 100000,
        "period": 60,  # 100000 requests per minute
        "burst": 10000
    }
    
    @classmethod
    def get_tier(cls, user: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """Get rate limit tier for user."""
        if not user:
            return cls.ANONYMOUS
        
        if user.get("is_admin"):
            return cls.ADMIN
        
        tier = user.get("rate_limit_tier", "free").upper()
        return getattr(cls, tier, cls.FREE)


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """
    Distributed rate limiter with Redis backend.
    
    Features:
    - Multiple rate limiting strategies
    - Per-user and per-IP limits
    - Burst handling
    - Comprehensive monitoring
    - Graceful degradation
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.strategies = {
            "fixed_window": FixedWindowStrategy(),
            "sliding_window": SlidingWindowStrategy(),
            "token_bucket": TokenBucketStrategy()
        }
        self.default_strategy = "sliding_window"
        
        # Statistics
        self.stats = {
            "total_checks": 0,
            "total_allowed": 0,
            "total_blocked": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def initialize(self):
        """Initialize Redis connection."""
        if not self.redis_client:
            self.redis_client = await redis.from_url(
                settings.redis_url,
                max_connections=settings.redis.max_connections,
                decode_responses=False,
                socket_timeout=settings.redis.socket_timeout,
                socket_connect_timeout=settings.redis.socket_connect_timeout,
                retry_on_timeout=settings.redis.retry_on_timeout,
                health_check_interval=settings.redis.health_check_interval
            )
            logger.info("Rate limiter initialized with Redis")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            logger.info("Rate limiter closed")
    
    async def check_rate_limit(
        self,
        key: str,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        strategy: str = None,
        user: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, int, int, int]:
        """
        Check if request is within rate limit.
        
        Args:
            key: Unique identifier for the client
            limit: Maximum requests per window
            window: Time window in seconds
            strategy: Rate limiting strategy
            user: User object for tier-based limits
        
        Returns:
            Tuple of (is_allowed, limit, remaining, reset_time)
        """
        await self.initialize()
        
        self.stats["total_checks"] += 1
        
        # Get tier-based limits if not specified
        if limit is None or window is None:
            tier = RateLimitTier.get_tier(user)
            limit = limit or tier["requests"]
            window = window or tier["period"]
        
        # Get strategy
        strategy_name = strategy or self.default_strategy
        strategy_impl = self.strategies.get(strategy_name)
        
        if not strategy_impl:
            logger.warning(f"Unknown rate limit strategy: {strategy_name}, using sliding_window")
            strategy_impl = self.strategies["sliding_window"]
        
        try:
            # Check cache first
            cache_key = f"rl:check:{key}:{limit}:{window}:{strategy_name}"
            cached = await self.redis_client.get(cache_key)
            
            if cached:
                self.stats["cache_hits"] += 1
                is_allowed, remaining, reset_time = eval(cached.decode())
                return is_allowed, limit, remaining, reset_time
            
            self.stats["cache_misses"] += 1
            
            # Perform rate limit check
            is_allowed, remaining, reset_time = await strategy_impl.check(
                self.redis_client,
                f"rl:{key}",
                limit,
                window
            )
            
            # Cache result for 1 second
            await self.redis_client.setex(
                cache_key,
                1,
                str((is_allowed, remaining, reset_time))
            )
            
            # Update statistics
            if is_allowed:
                self.stats["total_allowed"] += 1
            else:
                self.stats["total_blocked"] += 1
            
            # Log rate limit event
            if not is_allowed:
                logger.warning(
                    "rate_limit_exceeded",
                    key=key[:20] + "..." if len(key) > 20 else key,
                    limit=limit,
                    window=window,
                    reset_time=reset_time
                )
            
            return is_allowed, limit, remaining, reset_time
            
        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiter: {e}")
            # Fail open - allow request when Redis is down
            return True, limit, limit - 1, int(time.time() + window)
        
        except Exception as e:
            logger.error(f"Unexpected error in rate limiter: {e}")
            # Fail safe - block request on unexpected errors
            return False, limit, 0, int(time.time() + window)
    
    async def get_remaining(
        self,
        key: str,
        limit: int,
        window: int,
        strategy: str = None
    ) -> int:
        """Get remaining requests for a key."""
        await self.initialize()
        
        strategy_name = strategy or self.default_strategy
        strategy_impl = self.strategies.get(strategy_name)
        
        if not strategy_impl:
            return limit
        
        try:
            if strategy_name == "fixed_window":
                current = int(time.time())
                window_key = f"rl:{key}:{current // window}"
                count = await self.redis_client.get(window_key)
                count = int(count) if count else 0
                return max(0, limit - count)
            
            elif strategy_name == "sliding_window":
                now = time.time()
                window_start = now - window
                count = await self.redis_client.zcount(f"rl:{key}", window_start, now)
                return max(0, limit - count)
            
            elif strategy_name == "token_bucket":
                bucket_key = f"rl:{key}:bucket"
                data = await self.redis_client.hgetall(bucket_key)
                if data:
                    tokens = float(data.get(b'tokens', limit))
                    return int(tokens)
                return limit
            
            return limit
            
        except Exception as e:
            logger.error(f"Error getting remaining: {e}")
            return limit
    
    async def reset_limit(self, key: str, strategy: str = None):
        """Reset rate limit for a key."""
        await self.initialize()
        
        try:
            if strategy == "fixed_window":
                current = int(time.time())
                window_key = f"rl:{key}:{current // 60}"
                await self.redis_client.delete(window_key)
            
            elif strategy == "sliding_window":
                await self.redis_client.delete(f"rl:{key}")
            
            elif strategy == "token_bucket":
                await self.redis_client.delete(f"rl:{key}:bucket")
            
            else:
                # Reset all strategies
                pattern = f"rl:{key}*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            
            logger.info(f"Rate limit reset for key: {key[:20]}...")
            
        except Exception as e:
            logger.error(f"Error resetting rate limit: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            **self.stats,
            "cache_hit_rate": (
                self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
                if self.stats["cache_hits"] + self.stats["cache_misses"] > 0
                else 0
            ),
            "block_rate": (
                self.stats["total_blocked"] / self.stats["total_checks"]
                if self.stats["total_checks"] > 0
                else 0
            )
        }


# ============================================================================
# RATE LIMIT DECORATOR
# ============================================================================

def rate_limit(
    limit: Optional[int] = None,
    window: Optional[int] = None,
    strategy: str = None,
    key_func: Optional[Callable] = None,
    error_message: str = "Rate limit exceeded. Please try again later."
):
    """
    Rate limiting decorator for endpoints.
    
    Args:
        limit: Maximum requests per window
        window: Time window in seconds
        strategy: Rate limiting strategy
        key_func: Function to generate rate limit key
        error_message: Custom error message
    
    Example:
        @app.get("/api/endpoint")
        @rate_limit(limit=100, window=60)
        async def endpoint():
            return {"message": "success"}
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                for _, value in kwargs.items():
                    if isinstance(value, Request):
                        request = value
                        break
            
            if not request:
                return await func(*args, **kwargs)
            
            # Generate rate limit key
            if key_func:
                key = key_func(request)
            else:
                # Try to get authenticated user
                user = None
                try:
                    user = await get_current_user_required()
                except Exception:
                    pass
                
                api_key = request.headers.get("X-API-Key")
                user_id = user.get("id") if user else None
                key = get_rate_limit_key(request, user_id, api_key)
            
            # Check rate limit
            limiter = RateLimiter()
            is_allowed, limit_actual, remaining, reset_time = await limiter.check_rate_limit(
                key=key,
                limit=limit,
                window=window,
                strategy=strategy,
                user=user
            )
            
            if not is_allowed:
                raise RateLimitExceededError(
                    limit=limit_actual,
                    reset_time=reset_time,
                    detail=error_message
                )
            
            response = await func(*args, **kwargs)
            
            # Add rate limit headers to response
            if hasattr(response, "headers"):
                response.headers["X-RateLimit-Limit"] = str(limit_actual)
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                response.headers["X-RateLimit-Reset"] = str(reset_time)
            
            return response
        
        return wrapper
    
    return decorator


# ============================================================================
# RATE LIMIT DEPENDENCY
# ============================================================================

class RateLimitDependency:
    """
    FastAPI dependency for rate limiting.
    
    Example:
        @app.get("/api/endpoint")
        async def endpoint(_: None = Depends(RateLimitDependency(limit=100, window=60))):
            return {"message": "success"}
    """
    
    def __init__(
        self,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        strategy: str = None,
        key_func: Optional[Callable] = None,
        error_message: str = "Rate limit exceeded. Please try again later."
    ):
        self.limit = limit
        self.window = window
        self.strategy = strategy
        self.key_func = key_func
        self.error_message = error_message
    
    async def __call__(self, request: Request):
        # Generate rate limit key
        if self.key_func:
            key = self.key_func(request)
        else:
            # Try to get authenticated user
            user = None
            try:
                user = await get_current_user_required()
            except Exception:
                pass
            
            api_key = request.headers.get("X-API-Key")
            user_id = user.get("id") if user else None
            key = get_rate_limit_key(request, user_id, api_key)
        
        # Check rate limit
        limiter = RateLimiter()
        is_allowed, limit_actual, remaining, reset_time = await limiter.check_rate_limit(
            key=key,
            limit=self.limit,
            window=self.window,
            strategy=self.strategy,
            user=user
        )
        
        if not is_allowed:
            raise RateLimitExceededError(
                limit=limit_actual,
                reset_time=reset_time,
                detail=self.error_message
            )
        
        # Store rate limit info in request state
        request.state.rate_limit = {
            "limit": limit_actual,
            "remaining": remaining,
            "reset": reset_time
        }
        
        return {
            "limit": limit_actual,
            "remaining": remaining,
            "reset": reset_time
        }


# ============================================================================
# CONCURRENT REQUEST LIMITER
# ============================================================================

class ConcurrentRequestLimiter:
    """
    Limit concurrent requests using semaphores.
    
    Prevents resource exhaustion from too many simultaneous requests.
    """
    
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
        self.waiting_requests = 0
    
    async def acquire(self):
        """Acquire semaphore slot."""
        self.waiting_requests += 1
        await self.semaphore.acquire()
        self.waiting_requests -= 1
        self.active_requests += 1
    
    def release(self):
        """Release semaphore slot."""
        self.semaphore.release()
        self.active_requests -= 1
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def get_stats(self) -> Dict[str, int]:
        """Get current statistics."""
        return {
            "max_concurrent": self.max_concurrent,
            "active_requests": self.active_requests,
            "waiting_requests": self.waiting_requests,
            "available_slots": self.max_concurrent - self.active_requests
        }


# ============================================================================
# BURST HANDLING
# ============================================================================

class BurstRateLimiter:
    """
    Rate limiter with burst support.
    
    Allows short bursts of traffic above the normal limit,
    with a cooldown period.
    """
    
    def __init__(self, normal_limit: int, burst_limit: int, window: int):
        self.normal_limit = normal_limit
        self.burst_limit = burst_limit
        self.window = window
        self.base_limiter = RateLimiter()
    
    async def check_rate_limit(
        self,
        key: str,
        user: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, int, int, int]:
        """Check rate limit with burst support."""
        # Check normal limit
        normal_allowed, limit, remaining, reset = await self.base_limiter.check_rate_limit(
            key=f"{key}:normal",
            limit=self.normal_limit,
            window=self.window,
            user=user
        )
        
        if normal_allowed:
            return True, self.normal_limit, remaining, reset
        
        # Check burst limit
        burst_allowed, _, burst_remaining, burst_reset = await self.base_limiter.check_rate_limit(
            key=f"{key}:burst",
            limit=self.burst_limit,
            window=self.window // 2,  # Shorter window for burst
            user=user
        )
        
        if burst_allowed:
            return True, self.burst_limit, burst_remaining, burst_reset
        
        return False, self.normal_limit, 0, reset


# ============================================================================
# IP-BASED RATE LIMITING
# ============================================================================

class IPRateLimiter:
    """IP-based rate limiting for anonymous users."""
    
    def __init__(self):
        self.limiter = RateLimiter()
    
    async def check_rate_limit(
        self,
        request: Request,
        limit: int = 60,
        window: int = 60
    ) -> Tuple[bool, int, int, int]:
        """Check rate limit by IP address."""
        client_ip = request.client.host if request.client else "unknown"
        key = f"ip:{client_ip}"
        
        return await self.limiter.check_rate_limit(
            key=key,
            limit=limit,
            window=window
        )


# ============================================================================
# API KEY RATE LIMITING
# ============================================================================

class APIKeyRateLimiter:
    """API key based rate limiting."""
    
    def __init__(self):
        self.limiter = RateLimiter()
    
    async def check_rate_limit(
        self,
        api_key: str,
        limit: int = 1000,
        window: int = 60
    ) -> Tuple[bool, int, int, int]:
        """Check rate limit by API key."""
        # Hash the API key for privacy
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        key = f"apikey:{hashed_key}"
        
        return await self.limiter.check_rate_limit(
            key=key,
            limit=limit,
            window=window
        )


# ============================================================================
# USER RATE LIMITING
# ============================================================================

class UserRateLimiter:
    """User-based rate limiting."""
    
    def __init__(self):
        self.limiter = RateLimiter()
    
    async def check_rate_limit(
        self,
        user_id: str,
        user: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, int, int, int]:
        """Check rate limit by user ID."""
        key = f"user:{user_id}"
        
        # Get tier-based limits
        tier = RateLimitTier.get_tier(user)
        
        return await self.limiter.check_rate_limit(
            key=key,
            limit=tier["requests"],
            window=tier["period"],
            user=user
        )


# ============================================================================
# RATE LIMIT MIDDLEWARE
# ============================================================================

class RateLimitMiddleware:
    """
    FastAPI middleware for global rate limiting.
    
    Applies rate limits to all requests before they reach endpoints.
    """
    
    def __init__(
        self,
        anonymous_limit: int = 20,
        anonymous_window: int = 60,
        authenticated_limit: int = 100,
        authenticated_window: int = 60
    ):
        self.anonymous_limit = anonymous_limit
        self.anonymous_window = anonymous_window
        self.authenticated_limit = authenticated_limit
        self.authenticated_window = authenticated_window
        self.limiter = RateLimiter()
    
    async def __call__(self, request: Request, call_next):
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/metrics", "/"]:
            return await call_next(request)
        
        # Check if user is authenticated
        user = None
        try:
            user = await get_current_user_required()
        except Exception:
            pass
        
        # Get rate limit key
        api_key = request.headers.get("X-API-Key")
        user_id = user.get("id") if user else None
        key = get_rate_limit_key(request, user_id, api_key)
        
        # Get limits based on authentication
        if user:
            limit = self.authenticated_limit
            window = self.authenticated_window
        else:
            limit = self.anonymous_limit
            window = self.anonymous_window
        
        # Check rate limit
        is_allowed, limit_actual, remaining, reset_time = await self.limiter.check_rate_limit(
            key=key,
            limit=limit,
            window=window,
            user=user
        )
        
        if not is_allowed:
            raise RateLimitExceededError(
                limit=limit_actual,
                reset_time=reset_time,
                detail="Too many requests. Please try again later."
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limit_actual)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response


# ============================================================================
# RATE LIMIT EXPORTER
# ============================================================================

class RateLimitExporter:
    """
    Export rate limit metrics for monitoring.
    """
    
    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get Prometheus-compatible metrics."""
        stats = await self.limiter.get_stats()
        
        return {
            "rate_limit_checks_total": stats["total_checks"],
            "rate_limit_allowed_total": stats["total_allowed"],
            "rate_limit_blocked_total": stats["total_blocked"],
            "rate_limit_cache_hits_total": stats["cache_hits"],
            "rate_limit_cache_misses_total": stats["cache_misses"],
            "rate_limit_cache_hit_rate": stats["cache_hit_rate"],
            "rate_limit_block_rate": stats["block_rate"]
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_rate_limiter = None


async def get_rate_limiter() -> RateLimiter:
    """Get singleton rate limiter instance."""
    global _rate_limiter
    if not _rate_limiter:
        _rate_limiter = RateLimiter()
        await _rate_limiter.initialize()
    return _rate_limiter


# ============================================================================
# FASTAPI DEPENDENCY
# ============================================================================

async def rate_limit_dependency(
    limit: int = 60,
    window: int = 60,
    strategy: str = "sliding_window"
) -> Callable:
    """
    Factory for rate limit dependencies.
    
    Example:
        @app.get("/api/endpoint")
        async def endpoint(_: None = Depends(rate_limit_dependency(limit=100, window=60))):
            return {"message": "success"}
    """
    limiter = await get_rate_limiter()
    
    async def dependency(request: Request):
        # Get rate limit key
        user = None
        try:
            user = await get_current_user_required()
        except Exception:
            pass
        
        api_key = request.headers.get("X-API-Key")
        user_id = user.get("id") if user else None
        key = get_rate_limit_key(request, user_id, api_key)
        
        # Check rate limit
        is_allowed, limit_actual, remaining, reset_time = await limiter.check_rate_limit(
            key=key,
            limit=limit,
            window=window,
            strategy=strategy,
            user=user
        )
        
        if not is_allowed:
            raise RateLimitExceededError(
                limit=limit_actual,
                reset_time=reset_time
            )
        
        return {
            "limit": limit_actual,
            "remaining": remaining,
            "reset": reset_time
        }
    
    return dependency


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Main rate limiter
    "RateLimiter",
    "get_rate_limiter",
    
    # Strategies
    "RateLimitStrategy",
    "FixedWindowStrategy",
    "SlidingWindowStrategy",
    "TokenBucketStrategy",
    
    # Limiters
    "ConcurrentRequestLimiter",
    "BurstRateLimiter",
    "IPRateLimiter",
    "APIKeyRateLimiter",
    "UserRateLimiter",
    
    # Tiers
    "RateLimitTier",
    
    # FastAPI integration
    "rate_limit",
    "RateLimitDependency",
    "rate_limit_dependency",
    "RateLimitMiddleware",
    
    # Monitoring
    "RateLimitExporter",
]