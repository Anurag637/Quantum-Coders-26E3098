"""
Circuit Breaker Pattern - Production Ready
Prevents cascading failures, detects service degradation, and provides automatic recovery.
Implements state machine with three states: CLOSED, OPEN, HALF_OPEN.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from collections import deque
import traceback
import random

from core.logging import get_logger
from core.exceptions import CircuitBreakerError

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# CIRCUIT BREAKER STATES
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests flow through
    OPEN = "open"         # Failure threshold exceeded, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitTrippedReason(Enum):
    """Reasons for circuit tripping."""
    FAILURE_THRESHOLD = "failure_threshold_exceeded"
    SLOW_CALL_THRESHOLD = "slow_call_threshold_exceeded"
    CONSECUTIVE_FAILURES = "consecutive_failures_exceeded"
    MANUAL = "manual_override"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    TIMEOUT = "timeout_threshold_exceeded"


# ============================================================================
# CIRCUIT BREAKER CONFIGURATION
# ============================================================================

class CircuitBreakerConfig:
    """
    Configuration for circuit breaker behavior.
    
    Default values are optimized for most services.
    Adjust based on service characteristics.
    """
    
    def __init__(
        self,
        # Failure threshold
        failure_threshold: int = 5,           # Number of failures to open circuit
        failure_timeout: int = 60,            # Time window for counting failures (seconds)
        
        # Slow call threshold
        slow_call_threshold: int = 10,        # Number of slow calls to open circuit
        slow_call_duration_ms: int = 1000,    # Call duration considered slow (ms)
        
        # Recovery settings
        open_timeout: int = 30,               # Time in OPEN state before HALF_OPEN (seconds)
        half_open_success_threshold: int = 3,  # Successes in HALF_OPEN to close circuit
        half_open_failure_threshold: int = 1,  # Failures in HALF_OPEN to reopen circuit
        
        # Advanced settings
        consecutive_failure_threshold: int = 3,  # Consecutive failures to open circuit
        rolling_window_size: int = 100,        # Size of rolling window for metrics
        minimum_calls: int = 10,              # Minimum calls before evaluating thresholds
        should_trip_on_timeout: bool = True,   # Trip circuit on timeout
        should_trip_on_exception: bool = True, # Trip circuit on exception
        
        # Monitoring
        record_exceptions: bool = True,        # Record exception details
        record_stack_traces: bool = False,     # Record stack traces (expensive)
        
        # Fallback
        enable_fallback: bool = True,          # Enable fallback responses
        fallback_timeout: int = 5,            # Timeout for fallback (seconds)
    ):
        self.failure_threshold = failure_threshold
        self.failure_timeout = failure_timeout
        self.slow_call_threshold = slow_call_threshold
        self.slow_call_duration_ms = slow_call_duration_ms
        self.open_timeout = open_timeout
        self.half_open_success_threshold = half_open_success_threshold
        self.half_open_failure_threshold = half_open_failure_threshold
        self.consecutive_failure_threshold = consecutive_failure_threshold
        self.rolling_window_size = rolling_window_size
        self.minimum_calls = minimum_calls
        self.should_trip_on_timeout = should_trip_on_timeout
        self.should_trip_on_exception = should_trip_on_exception
        self.record_exceptions = record_exceptions
        self.record_stack_traces = record_stack_traces
        self.enable_fallback = enable_fallback
        self.fallback_timeout = fallback_timeout


# ============================================================================
# CIRCUIT BREAKER METRICS
# ============================================================================

class CircuitBreakerMetrics:
    """Metrics collector for circuit breaker."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.slow_calls = 0
        self.timeout_calls = 0
        self.rejected_calls = 0
        self.fallback_successes = 0
        self.fallback_failures = 0
        
        # Rolling window for recent calls
        self.recent_outcomes = deque(maxlen=self.config.rolling_window_size)
        
        # Exception tracking
        self.exception_counts = {}
        self.last_exception = None
        self.last_exception_time = None
        
        # State transitions
        self.state_transitions = []
        self.last_state_change = time.time()
        
        # Timing
        self.total_duration_ms = 0
        self.min_duration_ms = float('inf')
        self.max_duration_ms = 0
    
    def record_success(self, duration_ms: float):
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.recent_outcomes.append(('success', time.time(), duration_ms))
        
        self.total_duration_ms += duration_ms
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)
    
    def record_failure(self, error: Exception, duration_ms: float):
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.recent_outcomes.append(('failure', time.time(), duration_ms, str(error)))
        
        if self.config.record_exceptions:
            error_type = type(error).__name__
            self.exception_counts[error_type] = self.exception_counts.get(error_type, 0) + 1
            self.last_exception = error
            self.last_exception_time = time.time()
    
    def record_slow_call(self, duration_ms: float):
        """Record a slow call."""
        self.slow_calls += 1
    
    def record_timeout(self, duration_ms: float):
        """Record a timeout."""
        self.timeout_calls += 1
    
    def record_rejected(self):
        """Record a rejected call (circuit open)."""
        self.rejected_calls += 1
        self.recent_outcomes.append(('rejected', time.time()))
    
    def record_fallback_success(self):
        """Record a successful fallback."""
        self.fallback_successes += 1
    
    def record_fallback_failure(self):
        """Record a failed fallback."""
        self.fallback_failures += 1
    
    def record_state_transition(self, from_state: CircuitState, to_state: CircuitState, reason: str):
        """Record a state transition."""
        self.state_transitions.append({
            'from': from_state.value,
            'to': to_state.value,
            'reason': reason,
            'timestamp': time.time()
        })
        self.last_state_change = time.time()
    
    def get_success_rate(self, window_seconds: int = None) -> float:
        """Get success rate within time window."""
        if window_seconds:
            cutoff = time.time() - window_seconds
            recent = [o for o in self.recent_outcomes if o[1] > cutoff]
            successes = sum(1 for o in recent if o[0] == 'success')
            total = len(recent)
            return successes / total if total > 0 else 1.0
        else:
            return self.successful_calls / self.total_calls if self.total_calls > 0 else 1.0
    
    def get_failure_rate(self, window_seconds: int = None) -> float:
        """Get failure rate within time window."""
        return 1 - self.get_success_rate(window_seconds)
    
    def get_average_duration_ms(self) -> float:
        """Get average call duration."""
        return self.total_duration_ms / self.successful_calls if self.successful_calls > 0 else 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "slow_calls": self.slow_calls,
            "timeout_calls": self.timeout_calls,
            "rejected_calls": self.rejected_calls,
            "fallback_successes": self.fallback_successes,
            "fallback_failures": self.fallback_failures,
            "success_rate": self.get_success_rate(),
            "failure_rate": self.get_failure_rate(),
            "success_rate_1m": self.get_success_rate(60),
            "success_rate_5m": self.get_success_rate(300),
            "success_rate_15m": self.get_success_rate(900),
            "average_duration_ms": self.get_average_duration_ms(),
            "min_duration_ms": self.min_duration_ms if self.min_duration_ms != float('inf') else 0,
            "max_duration_ms": self.max_duration_ms,
            "exception_counts": self.exception_counts,
            "last_exception": str(self.last_exception) if self.last_exception else None,
            "last_exception_time": self.last_exception_time,
            "state_transitions": self.state_transitions[-10:],  # Last 10 transitions
            "last_state_change": self.last_state_change
        }


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    State machine:
        CLOSED -> OPEN: Failure threshold exceeded
        OPEN -> HALF_OPEN: Timeout elapsed
        HALF_OPEN -> CLOSED: Success threshold met
        HALF_OPEN -> OPEN: Failure detected
    
    Features:
    - Thread-safe state management
    - Configurable thresholds
    - Metrics collection
    - Event callbacks
    - Automatic recovery
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback_func: Optional[Callable] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback_func = fallback_func
        
        # State
        self._state = CircuitState.CLOSED
        self._state_lock = asyncio.Lock()
        self._state_changed_at = time.time()
        self._trip_reason = None
        
        # Metrics
        self.metrics = CircuitBreakerMetrics(self.config)
        
        # Half-open state tracking
        self._half_open_successes = 0
        self._half_open_failures = 0
        
        # Callbacks
        self._on_state_change_callbacks = []
        self._on_trip_callbacks = []
        self._on_reset_callbacks = []
        self._on_reject_callbacks = []
        
        logger.info(
            "circuit_breaker_initialized",
            name=self.name,
            state=self._state.value
        )
    
    # ========================================================================
    # PROPERTIES
    # ========================================================================
    
    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open."""
        return self._state == CircuitState.HALF_OPEN
    
    @property
    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        return self._state != CircuitState.OPEN
    
    @property
    def time_in_current_state(self) -> float:
        """Get time spent in current state (seconds)."""
        return time.time() - self._state_changed_at
    
    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================
    
    async def _transition_to(self, new_state: CircuitState, reason: str):
        """Transition to a new state."""
        async with self._state_lock:
            old_state = self._state
            if old_state == new_state:
                return
            
            self._state = new_state
            self._state_changed_at = time.time()
            
            # Reset half-open counters
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_successes = 0
                self._half_open_failures = 0
            
            # Reset metrics on close
            if new_state == CircuitState.CLOSED:
                self.metrics.reset()
            
            # Record transition
            self.metrics.record_state_transition(old_state, new_state, reason)
            
            logger.info(
                "circuit_state_changed",
                name=self.name,
                from_state=old_state.value,
                to_state=new_state.value,
                reason=reason
            )
            
            # Execute callbacks
            for callback in self._on_state_change_callbacks:
                await self._call_callback(callback, old_state, new_state, reason)
    
    async def trip(self, reason: CircuitTrippedReason, details: Optional[str] = None):
        """Trip the circuit (CLOSED -> OPEN)."""
        if self._state != CircuitState.CLOSED:
            return
        
        self._trip_reason = {
            'reason': reason.value,
            'details': details,
            'timestamp': time.time()
        }
        
        await self._transition_to(
            CircuitState.OPEN,
            f"Circuit tripped: {reason.value} - {details}" if details else reason.value
        )
        
        # Execute trip callbacks
        for callback in self._on_trip_callbacks:
            await self._call_callback(callback, reason, details)
    
    async def reset(self):
        """Reset the circuit (any state -> CLOSED)."""
        await self._transition_to(
            CircuitState.CLOSED,
            "Manual reset"
        )
        
        # Execute reset callbacks
        for callback in self._on_reset_callbacks:
            await self._call_callback(callback)
    
    async def _check_state(self):
        """Check if state should transition based on conditions."""
        if self._state == CircuitState.OPEN:
            # Check if timeout elapsed
            if self.time_in_current_state >= self.config.open_timeout:
                await self._transition_to(
                    CircuitState.HALF_OPEN,
                    f"Open timeout elapsed ({self.config.open_timeout}s)"
                )
        
        elif self._state == CircuitState.CLOSED:
            # Check failure thresholds
            await self._evaluate_failure_conditions()
        
        elif self._state == CircuitState.HALF_OPEN:
            # Check success/failure thresholds
            if self._half_open_successes >= self.config.half_open_success_threshold:
                await self._transition_to(
                    CircuitState.CLOSED,
                    f"Success threshold met ({self._half_open_successes} successes)"
                )
            elif self._half_open_failures >= self.config.half_open_failure_threshold:
                await self.trip(
                    CircuitTrippedReason.CONSECUTIVE_FAILURES,
                    f"Failure in half-open state ({self._half_open_failures} failures)"
                )
    
    async def _evaluate_failure_conditions(self):
        """Evaluate if failure thresholds are exceeded."""
        # Need minimum calls to evaluate
        if self.metrics.total_calls < self.config.minimum_calls:
            return
        
        # Check failure rate in recent window
        recent_failures = sum(
            1 for o in self.metrics.recent_outcomes
            if o[0] == 'failure' and o[1] > time.time() - self.config.failure_timeout
        )
        
        if recent_failures >= self.config.failure_threshold:
            await self.trip(
                CircuitTrippedReason.FAILURE_THRESHOLD,
                f"{recent_failures} failures in {self.config.failure_timeout}s"
            )
            return
        
        # Check slow calls
        if self.metrics.slow_calls >= self.config.slow_call_threshold:
            await self.trip(
                CircuitTrippedReason.SLOW_CALL_THRESHOLD,
                f"{self.metrics.slow_calls} slow calls"
            )
            return
    
    # ========================================================================
    # EXECUTION
    # ========================================================================
    
    async def execute(
        self,
        func: Callable,
        *args,
        fallback_func: Optional[Callable] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            fallback_func: Optional fallback function
            timeout: Timeout in seconds
            **kwargs: Function keyword arguments
        
        Returns:
            Function result or fallback result
        
        Raises:
            CircuitBreakerError: Circuit is open and no fallback
            Exception: Original exception from func
        """
        # Check state before execution
        await self._check_state()
        
        if self._state == CircuitState.OPEN:
            self.metrics.record_rejected()
            logger.warning(
                "circuit_open_reject",
                name=self.name,
                time_open=self.time_in_current_state
            )
            
            # Execute fallback if available
            fallback = fallback_func or self.fallback_func
            if fallback and self.config.enable_fallback:
                try:
                    result = await asyncio.wait_for(
                        self._execute_fallback(fallback, *args, **kwargs),
                        timeout=self.config.fallback_timeout
                    )
                    self.metrics.record_fallback_success()
                    return result
                except Exception as e:
                    self.metrics.record_fallback_failure()
                    logger.error(
                        "fallback_failed",
                        name=self.name,
                        error=str(e)
                    )
            
            # No fallback or fallback failed
            raise CircuitBreakerError(
                service=self.name,
                retry_after=int(self.config.open_timeout - self.time_in_current_state)
            )
        
        # Execute function
        start_time = time.time()
        
        try:
            if timeout:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
            else:
                result = await func(*args, **kwargs)
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Record success
            self.metrics.record_success(duration_ms)
            
            # Check for slow call
            if duration_ms > self.config.slow_call_duration_ms:
                self.metrics.record_slow_call(duration_ms)
            
            # Handle half-open state
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
            
            return result
            
        except asyncio.TimeoutError as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_failure(e, duration_ms)
            self.metrics.record_timeout(duration_ms)
            
            if self.config.should_trip_on_timeout:
                await self.trip(
                    CircuitTrippedReason.TIMEOUT,
                    f"Timeout after {timeout}s"
                )
            
            raise
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_failure(e, duration_ms)
            
            if self.config.should_trip_on_exception:
                # Check consecutive failures
                recent_failures = [
                    o for o in list(self.metrics.recent_outcomes)[-self.config.consecutive_failure_threshold:]
                    if o[0] == 'failure'
                ]
                
                if len(recent_failures) >= self.config.consecutive_failure_threshold:
                    await self.trip(
                        CircuitTrippedReason.CONSECUTIVE_FAILURES,
                        f"{len(recent_failures)} consecutive failures"
                    )
            
            # Handle half-open state
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_failures += 1
            
            raise
    
    async def _execute_fallback(self, fallback_func: Callable, *args, **kwargs) -> Any:
        """Execute fallback function."""
        try:
            if asyncio.iscoroutinefunction(fallback_func):
                return await fallback_func(*args, **kwargs)
            else:
                return fallback_func(*args, **kwargs)
        except Exception as e:
            logger.error(
                "fallback_execution_failed",
                name=self.name,
                error=str(e)
            )
            raise
    
    # ========================================================================
    # CALLBACK REGISTRATION
    # ========================================================================
    
    def on_state_change(self, callback: Callable):
        """Register callback for state changes."""
        self._on_state_change_callbacks.append(callback)
        return self
    
    def on_trip(self, callback: Callable):
        """Register callback for circuit trip."""
        self._on_trip_callbacks.append(callback)
        return self
    
    def on_reset(self, callback: Callable):
        """Register callback for circuit reset."""
        self._on_reset_callbacks.append(callback)
        return self
    
    def on_reject(self, callback: Callable):
        """Register callback for rejected requests."""
        self._on_reject_callbacks.append(callback)
        return self
    
    async def _call_callback(self, callback: Callable, *args, **kwargs):
        """Execute a callback safely."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(
                "callback_execution_failed",
                name=self.name,
                callback=callback.__name__,
                error=str(e)
            )
    
    # ========================================================================
    # HEALTH & METRICS
    # ========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """Get health status of circuit breaker."""
        return {
            "name": self.name,
            "state": self._state.value,
            "is_available": self.is_available,
            "time_in_state": self.time_in_current_state,
            "trip_reason": self._trip_reason,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "failure_timeout": self.config.failure_timeout,
                "open_timeout": self.config.open_timeout,
                "half_open_success_threshold": self.config.half_open_success_threshold,
                "half_open_failure_threshold": self.config.half_open_failure_threshold,
                "minimum_calls": self.config.minimum_calls
            },
            "metrics": self.metrics.get_stats()
        }
    
    async def force_open(self, reason: str = "Manual override"):
        """Force circuit to OPEN state."""
        await self._transition_to(CircuitState.OPEN, reason)
        self._trip_reason = {
            'reason': CircuitTrippedReason.MANUAL.value,
            'details': reason,
            'timestamp': time.time()
        }
    
    async def force_close(self, reason: str = "Manual override"):
        """Force circuit to CLOSED state."""
        await self._transition_to(CircuitState.CLOSED, reason)
        self._trip_reason = None


# ============================================================================
# CIRCUIT BREAKER REGISTRY
# ============================================================================

class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Provides:
    - Centralized management
    - Shared configuration
    - Bulk operations
    - Monitoring
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()
        self._lock = asyncio.Lock()
        
        logger.info("circuit_breaker_registry_initialized")
    
    def get(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback_func: Optional[Callable] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                config=config or self._default_config,
                fallback_func=fallback_func
            )
        
        return self._breakers[name]
    
    def get_all(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers."""
        return self._breakers.copy()
    
    async def reset_all(self):
        """Reset all circuit breakers to CLOSED state."""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.reset()
        
        logger.info("all_circuit_breakers_reset")
    
    async def force_open_all(self, reason: str = "Manual override"):
        """Force all circuit breakers to OPEN state."""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.force_open(reason)
        
        logger.info(f"all_circuit_breakers_forced_open: {reason}")
    
    async def force_close_all(self, reason: str = "Manual override"):
        """Force all circuit breakers to CLOSED state."""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.force_close(reason)
        
        logger.info(f"all_circuit_breakers_forced_close: {reason}")
    
    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all circuit breakers."""
        return {
            name: breaker.health_check()
            for name, breaker in self._breakers.items()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all circuit breakers."""
        total = len(self._breakers)
        closed = sum(1 for b in self._breakers.values() if b.is_closed)
        open_count = sum(1 for b in self._breakers.values() if b.is_open)
        half_open = sum(1 for b in self._breakers.values() if b.is_half_open)
        
        total_rejected = sum(
            b.metrics.rejected_calls for b in self._breakers.values()
        )
        total_failures = sum(
            b.metrics.failed_calls for b in self._breakers.values()
        )
        
        return {
            "total_circuits": total,
            "closed": closed,
            "open": open_count,
            "half_open": half_open,
            "total_rejected_calls": total_rejected,
            "total_failed_calls": total_failures,
            "circuits": [
                {
                    "name": name,
                    "state": breaker.state.value,
                    "rejected": breaker.metrics.rejected_calls,
                    "failure_rate": breaker.metrics.get_failure_rate()
                }
                for name, breaker in self._breakers.items()
            ]
        }


# ============================================================================
# CIRCUIT BREAKER DECORATOR
# ============================================================================

def circuit_breaker(
    name: Optional[str] = None,
    config: Optional[CircuitBreakerConfig] = None,
    fallback_func: Optional[Callable] = None,
    timeout: Optional[float] = None
):
    """
    Circuit breaker decorator for functions.
    
    Args:
        name: Circuit breaker name (defaults to function name)
        config: Circuit breaker configuration
        fallback_func: Fallback function
        timeout: Timeout in seconds
    
    Example:
        @circuit_breaker(name="model_inference", timeout=30)
        async def call_model(prompt):
            return await model.generate(prompt)
    """
    def decorator(func):
        breaker_name = name or func.__name__
        registry = CircuitBreakerRegistry()
        breaker = registry.get(breaker_name, config, fallback_func)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.execute(
                func,
                *args,
                timeout=timeout,
                **kwargs
            )
        
        # Attach breaker to function for inspection
        wrapper.__circuit_breaker__ = breaker
        
        return wrapper
    
    return decorator


# ============================================================================
# BULKHEAD PATTERN
# ============================================================================

class Bulkhead:
    """
    Bulkhead pattern for isolating failures.
    
    Limits concurrent calls to prevent resource exhaustion.
    Similar to circuit breaker but for concurrency control.
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_queue_size: int = 100
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue = asyncio.Queue(maxsize=max_queue_size)
        self._active_calls = 0
        self._queued_calls = 0
        self._rejected_calls = 0
        self._lock = asyncio.Lock()
        
        logger.info(
            "bulkhead_initialized",
            name=self.name,
            max_concurrent=max_concurrent,
            max_queue_size=max_queue_size
        )
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead isolation."""
        # Try to acquire semaphore immediately
        if self._semaphore.locked():
            # Queue if space available
            async with self._lock:
                if self._queued_calls < self.max_queue_size:
                    self._queued_calls += 1
                else:
                    self._rejected_calls += 1
                    raise CircuitBreakerError(
                        service=self.name,
                        retry_after=5,
                        detail="Bulkhead queue full"
                    )
            
            # Wait in queue
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=30
                )
            except asyncio.TimeoutError:
                async with self._lock:
                    self._queued_calls -= 1
                raise CircuitBreakerError(
                    service=self.name,
                    retry_after=5,
                    detail="Bulkhead queue timeout"
                )
            
            async with self._lock:
                self._queued_calls -= 1
        else:
            await self._semaphore.acquire()
        
        # Execute function
        async with self._lock:
            self._active_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            self._semaphore.release()
            async with self._lock:
                self._active_calls -= 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get bulkhead statistics."""
        return {
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "active_calls": self._active_calls,
            "queued_calls": self._queued_calls,
            "rejected_calls": self._rejected_calls,
            "available_permits": self.max_concurrent - self._active_calls,
            "available_queue": self.max_queue_size - self._queued_calls
        }


# ============================================================================
# SINGLETON REGISTRY
# ============================================================================

_circuit_breaker_registry = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get singleton circuit breaker registry."""
    global _circuit_breaker_registry
    if not _circuit_breaker_registry:
        _circuit_breaker_registry = CircuitBreakerRegistry()
    return _circuit_breaker_registry


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "CircuitState",
    "CircuitTrippedReason",
    
    # Configuration
    "CircuitBreakerConfig",
    
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    
    # Decorator
    "circuit_breaker",
    
    # Bulkhead
    "Bulkhead",
]