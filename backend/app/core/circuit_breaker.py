import time
from enum import Enum
from typing import Callable, Awaitable, Dict


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Async circuit breaker implementation.
    Designed per model/backend instance.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_success_threshold: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_success_threshold = half_open_success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float = 0.0

    async def call(self, func: Callable[..., Awaitable], *args, **kwargs):
        """
        Execute protected async function.
        """

        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit is open. Backend temporarily disabled.")

        try:
            result = await func(*args, **kwargs)

            await self._on_success()
            return result

        except Exception as e:
            await self._on_failure()
            raise e

    async def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_success_threshold:
                self._reset()
        else:
            self._reset()

    async def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def _reset(self):
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED


# ===============================
# Circuit Breaker Registry
# ===============================

class CircuitBreakerRegistry:
    """
    Maintains per-model circuit breakers.
    """

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get(self, model_name: str) -> CircuitBreaker:
        if model_name not in self._breakers:
            self._breakers[model_name] = CircuitBreaker()
        return self._breakers[model_name]


# Global registry instance
circuit_registry = CircuitBreakerRegistry()