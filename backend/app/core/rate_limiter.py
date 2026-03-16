import time
from typing import Optional

import redis.asyncio as redis
from fastapi import HTTPException, status

from backend.app.core.config import get_settings


settings = get_settings()


class RateLimiter:
    """
    Redis-based sliding window rate limiter.
    """

    def __init__(self):
        self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.max_requests = settings.RATE_LIMIT_REQUESTS
        self.window_seconds = settings.RATE_LIMIT_WINDOW_SECONDS

    async def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed for given identifier.
        """

        current_time = int(time.time())
        window_start = current_time - self.window_seconds

        key = f"rate_limit:{identifier}"

        # Remove old entries
        await self.redis.zremrangebyscore(key, 0, window_start)

        # Count requests in window
        request_count = await self.redis.zcard(key)

        if request_count >= self.max_requests:
            return False

        # Add current request timestamp
        await self.redis.zadd(key, {str(current_time): current_time})
        await self.redis.expire(key, self.window_seconds)

        return True


# ===============================
# FastAPI Dependency
# ===============================

rate_limiter = RateLimiter()


async def enforce_rate_limit(identifier: str):
    """
    Raises HTTPException if rate limit exceeded.
    """

    allowed = await rate_limiter.is_allowed(identifier)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        )