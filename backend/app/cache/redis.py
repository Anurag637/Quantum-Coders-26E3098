import logging
import redis.asyncio as redis
from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

redis_client = None


class FakeRedis:
    """
    In-memory Redis stub used when a real Redis server is unavailable.
    Implements only the methods used by the application (get, set, setex).
    Cache is NOT shared between processes and is NOT persistent — suitable
    for local development only.
    """

    def __init__(self):
        self._store: dict = {}

    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value: str):
        self._store[key] = value

    async def setex(self, key: str, ttl_seconds: int, value: str):
        # TTL is intentionally ignored in the stub — fine for dev
        self._store[key] = value

    async def ping(self):
        return True


async def init_redis():
    global redis_client
    try:
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        await client.ping()
        redis_client = client
        logger.info("Connected to Redis at %s", settings.REDIS_URL)
    except Exception as exc:
        logger.warning(
            "Redis unavailable (%s). Falling back to in-memory cache. "
            "Caching will not persist across restarts.",
            exc,
        )
        redis_client = FakeRedis()


async def close_redis():
    global redis_client
    if redis_client and not isinstance(redis_client, FakeRedis):
        await redis_client.aclose()


async def get_redis():
    global redis_client
    if redis_client is None:
        await init_redis()
    return redis_client
