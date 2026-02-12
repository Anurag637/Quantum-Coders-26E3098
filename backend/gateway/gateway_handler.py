"""
Gateway Handler - Production Ready
Core request processing pipeline for LLM inference with intelligent routing,
caching, monitoring, and fault tolerance.
"""

import time
import uuid
import asyncio
from typing import Dict, Any, Optional, List, AsyncGenerator, Tuple
from datetime import datetime
import hashlib
import json

from fastapi import Request

from core.logging import get_logger, get_request_logger, get_model_logger
from core.exceptions import (
    ModelNotAvailableError,
    InvalidPromptError,
    RoutingError,
    RateLimitExceededError,
    CircuitBreakerError
)
from core.circuit_breaker import CircuitBreakerRegistry, CircuitBreakerConfig
from config import settings
from models.model_manager import ModelManager
from models.model_registry import ModelRegistry
from router.prompt_router import PromptRouter
from router.prompt_analyzer import PromptAnalyzer
from cache.cache_manager import CacheManager
from cache.semantic_cache import SemanticCache
from monitoring.metrics import MetricsCollector
from database.repositories.request_log_repository import RequestLogRepository
from database.repositories.routing_repository import RoutingRepository
from services.token_counter import TokenCounter
from services.cost_calculator import CostCalculator

# Initialize loggers
logger = get_logger(__name__)
request_logger = get_request_logger()
model_logger = get_model_logger()

# ============================================================================
# GATEWAY CONFIGURATION
# ============================================================================

class GatewayConfig:
    """Configuration for gateway handler."""
    
    def __init__(self):
        # Request processing
        self.max_prompt_length = 10000
        self.max_tokens = 4096
        self.default_temperature = 0.7
        self.default_top_p = 0.95
        
        # Timeouts
        self.request_timeout = 60
        self.streaming_timeout = 120
        self.model_load_timeout = 300
        
        # Caching
        self.enable_cache = settings.cache.enabled
        self.enable_semantic_cache = settings.cache.semantic_enabled
        self.cache_ttl = settings.cache.default_ttl
        self.semantic_threshold = settings.cache.similarity_threshold
        
        # Circuit breakers
        self.enable_circuit_breakers = settings.enable_circuit_breaker
        self.model_circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            failure_timeout=60,
            open_timeout=30,
            half_open_success_threshold=3,
            consecutive_failure_threshold=3,
            minimum_calls=10
        )
        
        self.external_api_circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            failure_timeout=30,
            open_timeout=60,
            half_open_success_threshold=2,
            consecutive_failure_threshold=2,
            minimum_calls=5
        )
        
        # Fallbacks
        self.enable_fallbacks = settings.enable_fallbacks
        self.default_fallback_model = "mistral-7b-instruct"
        
        # Monitoring
        self.record_request_logs = True
        self.record_metrics = True


# ============================================================================
# GATEWAY HANDLER
# ============================================================================

class GatewayHandler:
    """
    Main gateway handler for LLM inference requests.
    
    Implements the complete request processing pipeline:
    1. Request validation
    2. Cache check (exact + semantic)
    3. Prompt analysis
    4. Model routing
    5. Load balancing
    6. Model inference
    7. Response handling
    8. Cache update
    9. Monitoring & logging
    """
    
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        model_registry: Optional[ModelRegistry] = None,
        prompt_router: Optional[PromptRouter] = None,
        prompt_analyzer: Optional[PromptAnalyzer] = None,
        cache_manager: Optional[CacheManager] = None,
        semantic_cache: Optional[SemanticCache] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        # Initialize components
        self.model_manager = model_manager or ModelManager()
        self.model_registry = model_registry or ModelRegistry()
        self.prompt_router = prompt_router or PromptRouter()
        self.prompt_analyzer = prompt_analyzer or PromptAnalyzer()
        self.cache_manager = cache_manager or CacheManager()
        self.semantic_cache = semantic_cache or SemanticCache()
        self.metrics_collector = metrics_collector or MetricsCollector()
        
        # Repositories
        self.request_log_repo = RequestLogRepository()
        self.routing_repo = RoutingRepository()
        
        # Utilities
        self.token_counter = TokenCounter()
        self.cost_calculator = CostCalculator()
        
        # Configuration
        self.config = GatewayConfig()
        
        # Circuit breakers
        self.circuit_breaker_registry = CircuitBreakerRegistry()
        
        logger.info(
            "gateway_handler_initialized",
            cache_enabled=self.config.enable_cache,
            semantic_cache_enabled=self.config.enable_semantic_cache,
            circuit_breakers_enabled=self.config.enable_circuit_breakers,
            fallbacks_enabled=self.config.enable_fallbacks
        )
    
    # ========================================================================
    # MAIN REQUEST PROCESSING PIPELINE
    # ========================================================================
    
    async def process_request(
        self,
        request_id: str,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        n: int = 1,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        user: Optional[str] = None,
        stream: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main request processing pipeline.
        
        Flow:
        1. Validate request
        2. Check cache
        3. Analyze prompt
        4. Route to model
        5. Execute inference
        6. Process response
        7. Update cache
        8. Log and record metrics
        """
        pipeline_start = time.time()
        
        logger.info(
            "gateway_request_started",
            request_id=request_id,
            prompt_length=len(prompt),
            model_requested=model,
            stream=stream,
            user=user
        )
        
        try:
            # ====================================================================
            # STEP 1: Validate request
            # ====================================================================
            await self._validate_request(prompt, max_tokens, temperature)
            
            # ====================================================================
            # STEP 2: Check cache
            # ====================================================================
            cache_result = await self._check_cache(
                request_id=request_id,
                prompt=prompt,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if cache_result.get("hit"):
                pipeline_time = (time.time() - pipeline_start) * 1000
                
                logger.info(
                    "gateway_cache_hit",
                    request_id=request_id,
                    cache_type=cache_result.get("cache_type"),
                    similarity=cache_result.get("similarity"),
                    pipeline_time_ms=round(pipeline_time, 2)
                )
                
                # Record metrics
                if self.config.record_metrics:
                    await self.metrics_collector.record_cache_hit(
                        cache_type=cache_result.get("cache_type"),
                        latency_ms=pipeline_time
                    )
                
                # Log request
                if self.config.record_request_logs:
                    await self._log_request(
                        request_id=request_id,
                        prompt=prompt,
                        response=cache_result["response"],
                        model=cache_result["model"],
                        tokens=cache_result.get("tokens", 0),
                        latency_ms=pipeline_time,
                        cache_hit=True,
                        cache_type=cache_result.get("cache_type"),
                        user_id=user
                    )
                
                return {
                    "id": f"chatcmpl-{request_id[:8]}",
                    "content": cache_result["response"],
                    "model": cache_result["model"],
                    "tokens": cache_result.get("tokens", 0),
                    "finish_reason": "stop",
                    "cache_hit": True,
                    "cache_type": cache_result.get("cache_type"),
                    "latency_ms": round(pipeline_time, 2)
                }
            
            # ====================================================================
            # STEP 3: Analyze prompt
            # ====================================================================
            analysis_start = time.time()
            analysis = await self.prompt_analyzer.analyze(prompt)
            analysis_time = (time.time() - analysis_start) * 1000
            
            # ====================================================================
            # STEP 4: Route to model
            # ====================================================================
            routing_start = time.time()
            
            # If model specified, validate and use it
            if model:
                selected_model = await self._validate_model(model)
                routing_strategy = "explicit"
                reasoning = f"Model explicitly requested by user: {model}"
                confidence = 1.0
            else:
                # Intelligent routing
                routing_decision = await self.prompt_router.make_decision(
                    prompt=prompt,
                    analysis=analysis,
                    user=user,
                    metadata=metadata
                )
                selected_model = routing_decision["selected_model"]
                routing_strategy = routing_decision["strategy"]
                reasoning = routing_decision["reasoning"]
                confidence = routing_decision["confidence"]
            
            routing_time = (time.time() - routing_start) * 1000
            
            # ====================================================================
            # STEP 5: Execute inference with circuit breaker
            # ====================================================================
            inference_start = time.time()
            
            # Get or create circuit breaker for this model
            if self.config.enable_circuit_breakers:
                if selected_model.startswith(("grok", "gpt", "claude", "command")):
                    # External API - more aggressive circuit breaker
                    circuit_config = self.config.external_api_circuit_config
                else:
                    # Local model - less aggressive circuit breaker
                    circuit_config = self.config.model_circuit_config
                
                circuit_breaker = self.circuit_breaker_registry.get(
                    name=f"model:{selected_model}",
                    config=circuit_config
                )
                
                # Execute with circuit breaker
                try:
                    inference_result = await circuit_breaker.execute(
                        self._execute_inference,
                        model_id=selected_model,
                        prompt=prompt,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=n,
                        stop=stop,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        timeout=self.config.request_timeout
                    )
                except CircuitBreakerError:
                    # Circuit is open - try fallback
                    logger.warning(
                        "circuit_breaker_open",
                        request_id=request_id,
                        model=selected_model
                    )
                    
                    if self.config.enable_fallbacks:
                        fallback_model = await self._get_fallback_model(selected_model)
                        logger.info(
                            "using_fallback_model",
                            request_id=request_id,
                            original_model=selected_model,
                            fallback_model=fallback_model
                        )
                        
                        inference_result = await self._execute_inference(
                            model_id=fallback_model,
                            prompt=prompt,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            n=n,
                            stop=stop,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty
                        )
                        
                        # Update routing info
                        selected_model = fallback_model
                        reasoning += f" (fallback from {selected_model} due to circuit breaker)"
                    else:
                        raise
            else:
                # Execute without circuit breaker
                inference_result = await self._execute_inference(
                    model_id=selected_model,
                    prompt=prompt,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty
                )
            
            inference_time = (time.time() - inference_start) * 1000
            
            # ====================================================================
            # STEP 6: Process response
            # ====================================================================
            response_content = inference_result["content"]
            
            # Calculate token usage
            prompt_tokens = self.token_counter.count_tokens(prompt)
            completion_tokens = self.token_counter.count_tokens(response_content)
            total_tokens = prompt_tokens + completion_tokens
            
            # Calculate cost
            cost = self.cost_calculator.calculate_cost(
                model=selected_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
            # ====================================================================
            # STEP 7: Update cache
            # ====================================================================
            if self.config.enable_cache and not cache_result.get("hit"):
                await self._update_cache(
                    prompt=prompt,
                    messages=messages,
                    response=response_content,
                    model=selected_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tokens=total_tokens
                )
            
            # ====================================================================
            # STEP 8: Log request and record metrics
            # ====================================================================
            pipeline_time = (time.time() - pipeline_start) * 1000
            
            if self.config.record_request_logs:
                await self._log_request(
                    request_id=request_id,
                    prompt=prompt,
                    response=response_content,
                    model=selected_model,
                    tokens=total_tokens,
                    latency_ms=pipeline_time,
                    cache_hit=False,
                    user_id=user,
                    metadata={
                        "analysis_time_ms": round(analysis_time, 2),
                        "routing_time_ms": round(routing_time, 2),
                        "inference_time_ms": round(inference_time, 2),
                        "routing_strategy": routing_strategy,
                        "confidence": confidence,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cost_usd": cost
                    }
                )
            
            # Log routing decision
            await self.routing_repo.log_decision({
                "request_id": request_id,
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
                "prompt_length": len(prompt),
                "prompt_type": analysis.get("prompt_type", "unknown"),
                "complexity_score": analysis.get("complexity_score", 0.5),
                "selected_model": selected_model,
                "routing_strategy": routing_strategy,
                "confidence_score": confidence,
                "reasoning": reasoning,
                "analysis_time_ms": round(analysis_time, 2),
                "decision_time_ms": round(routing_time, 2),
                "total_time_ms": round(pipeline_time, 2),
                "cache_hit": False
            })
            
            if self.config.record_metrics:
                await self.metrics_collector.record_inference(
                    model_id=selected_model,
                    latency_ms=inference_time,
                    tokens=total_tokens,
                    cost=cost,
                    success=True
                )
            
            logger.info(
                "gateway_request_completed",
                request_id=request_id,
                model=selected_model,
                tokens=total_tokens,
                latency_ms=round(pipeline_time, 2),
                cache_hit=False
            )
            
            # ====================================================================
            # STEP 9: Return response
            # ====================================================================
            return {
                "id": f"chatcmpl-{request_id[:8]}",
                "content": response_content,
                "model": selected_model,
                "tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": inference_result.get("finish_reason", "stop"),
                "cache_hit": False,
                "latency_ms": round(pipeline_time, 2),
                "analysis_time_ms": round(analysis_time, 2),
                "routing_time_ms": round(routing_time, 2),
                "inference_time_ms": round(inference_time, 2),
                "routing_strategy": routing_strategy,
                "confidence": confidence,
                "reasoning": reasoning,
                "cost_usd": round(cost, 6)
            }
            
        except InvalidPromptError as e:
            logger.error(
                "gateway_invalid_prompt",
                request_id=request_id,
                error=str(e)
            )
            raise
            
        except ModelNotAvailableError as e:
            logger.error(
                "gateway_model_not_available",
                request_id=request_id,
                model=model,
                error=str(e)
            )
            
            if self.config.record_metrics:
                await self.metrics_collector.record_inference(
                    model_id=model or "unknown",
                    latency_ms=0,
                    tokens=0,
                    cost=0,
                    success=False,
                    error=str(e)
                )
            
            raise
            
        except RoutingError as e:
            logger.error(
                "gateway_routing_failed",
                request_id=request_id,
                error=str(e)
            )
            raise
            
        except Exception as e:
            logger.error(
                "gateway_request_failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            
            if self.config.record_metrics:
                await self.metrics_collector.record_inference(
                    model_id=model or "unknown",
                    latency_ms=0,
                    tokens=0,
                    cost=0,
                    success=False,
                    error=str(e)
                )
            
            raise
    
    # ========================================================================
    # STREAMING REQUEST PROCESSING
    # ========================================================================
    
    async def process_streaming_request(
        self,
        request_id: str,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        user: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process streaming inference request.
        
        Yields tokens as they are generated.
        """
        pipeline_start = time.time()
        
        logger.info(
            "gateway_streaming_started",
            request_id=request_id,
            prompt_length=len(prompt),
            model_requested=model
        )
        
        try:
            # ====================================================================
            # STEP 1: Validate request
            # ====================================================================
            await self._validate_request(prompt, max_tokens, temperature)
            
            # ====================================================================
            # STEP 2: Analyze prompt
            # ====================================================================
            analysis = await self.prompt_analyzer.analyze(prompt)
            
            # ====================================================================
            # STEP 3: Route to model
            # ====================================================================
            if model:
                selected_model = await self._validate_model(model)
                routing_strategy = "explicit"
                reasoning = f"Model explicitly requested by user: {model}"
            else:
                routing_decision = await self.prompt_router.make_decision(
                    prompt=prompt,
                    analysis=analysis,
                    user=user,
                    metadata=metadata
                )
                selected_model = routing_decision["selected_model"]
                routing_strategy = routing_decision["strategy"]
                reasoning = routing_decision["reasoning"]
            
            # ====================================================================
            # STEP 4: Execute streaming inference
            # ====================================================================
            model_backend = await self.model_manager.get_backend(selected_model)
            
            full_response = ""
            token_count = 0
            
            async for chunk in model_backend.generate_stream(
                prompt=prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            ):
                full_response += chunk
                token_count += 1
                
                yield {
                    "id": f"chatcmpl-{request_id[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": selected_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
            
            # Send final chunk with finish reason
            yield {
                "id": f"chatcmpl-{request_id[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": selected_model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            
            # ====================================================================
            # STEP 5: Log request and record metrics
            # ====================================================================
            pipeline_time = (time.time() - pipeline_start) * 1000
            
            # Calculate token usage
            prompt_tokens = self.token_counter.count_tokens(prompt)
            completion_tokens = self.token_counter.count_tokens(full_response)
            total_tokens = prompt_tokens + completion_tokens
            
            # Calculate cost
            cost = self.cost_calculator.calculate_cost(
                model=selected_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
            if self.config.record_request_logs:
                await self._log_request(
                    request_id=request_id,
                    prompt=prompt,
                    response=full_response,
                    model=selected_model,
                    tokens=total_tokens,
                    latency_ms=pipeline_time,
                    cache_hit=False,
                    user_id=user,
                    metadata={
                        "streaming": True,
                        "tokens_per_second": token_count / (pipeline_time / 1000),
                        "cost_usd": cost
                    }
                )
            
            if self.config.record_metrics:
                await self.metrics_collector.record_inference(
                    model_id=selected_model,
                    latency_ms=pipeline_time,
                    tokens=total_tokens,
                    cost=cost,
                    success=True,
                    streaming=True
                )
            
            logger.info(
                "gateway_streaming_completed",
                request_id=request_id,
                model=selected_model,
                tokens=total_tokens,
                latency_ms=round(pipeline_time, 2),
                tokens_per_second=round(token_count / (pipeline_time / 1000), 2)
            )
            
        except Exception as e:
            logger.error(
                "gateway_streaming_failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            
            # Send error chunk
            yield {
                "id": f"chatcmpl-{request_id[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model or "unknown",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error"
                }]
            }
            
            raise
    
    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================
    
    async def _validate_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> None:
        """Validate request parameters."""
        if not prompt or not prompt.strip():
            raise InvalidPromptError("Prompt cannot be empty")
        
        if len(prompt) > self.config.max_prompt_length:
            raise InvalidPromptError(
                f"Prompt length {len(prompt)} exceeds maximum {self.config.max_prompt_length}"
            )
        
        if max_tokens > self.config.max_tokens:
            raise InvalidPromptError(
                f"max_tokens {max_tokens} exceeds maximum {self.config.max_tokens}"
            )
        
        if temperature < 0 or temperature > 2:
            raise InvalidPromptError(
                f"temperature {temperature} must be between 0 and 2"
            )
    
    async def _validate_model(self, model_id: str) -> str:
        """Validate that model exists and is available."""
        model = await self.model_registry.get_model(model_id)
        
        if not model:
            raise ModelNotAvailableError(
                model_id=model_id,
                detail=f"Model '{model_id}' not found in registry"
            )
        
        if model.get("status") != "ready" and model.get("type") != "external":
            raise ModelNotAvailableError(
                model_id=model_id,
                status=model.get("status"),
                detail=f"Model '{model_id}' is not ready (status: {model.get('status')})"
            )
        
        # For external models, check if API key is configured
        if model.get("type") == "external":
            provider = model.get("provider", "").lower()
            api_key_configured = False
            
            if "grok" in provider or "xai" in provider:
                api_key_configured = settings.grok_api_key_value is not None
            elif "openai" in provider:
                api_key_configured = settings.openai_api_key_value is not None
            elif "anthropic" in provider:
                api_key_configured = settings.anthropic_api_key_value is not None
            elif "cohere" in provider:
                api_key_configured = settings.cohere_api_key_value is not None
            
            if not api_key_configured:
                raise ModelNotAvailableError(
                    model_id=model_id,
                    detail=f"API key for {model.get('provider')} is not configured"
                )
        
        return model_id
    
    async def _check_cache(
        self,
        request_id: str,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        model: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Check cache for existing response."""
        if not self.config.enable_cache:
            return {"hit": False}
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            prompt=prompt,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Check exact cache
        cached = await self.cache_manager.get(cache_key)
        if cached:
            return {
                "hit": True,
                "cache_type": "exact",
                "response": cached["response"],
                "model": cached["model"],
                "tokens": cached.get("tokens"),
                "similarity": 1.0
            }
        
        # Check semantic cache if enabled
        if self.config.enable_semantic_cache:
            semantic_result = await self.semantic_cache.get(
                prompt=prompt,
                threshold=self.config.semantic_threshold
            )
            
            if semantic_result["hit"]:
                return {
                    "hit": True,
                    "cache_type": "semantic",
                    "response": semantic_result["response"],
                    "model": semantic_result["model"],
                    "tokens": semantic_result.get("tokens"),
                    "similarity": semantic_result["similarity"]
                }
        
        return {"hit": False}
    
    async def _update_cache(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        response: str,
        model: str,
        temperature: float,
        max_tokens: int,
        tokens: int
    ) -> None:
        """Update cache with response."""
        cache_key = self._generate_cache_key(
            prompt=prompt,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        cache_data = {
            "response": response,
            "model": model,
            "tokens": tokens,
            "timestamp": time.time()
        }
        
        await self.cache_manager.set(
            key=cache_key,
            value=cache_data,
            ttl=self.config.cache_ttl
        )
        
        # Update semantic cache
        if self.config.enable_semantic_cache:
            await self.semantic_cache.set(
                prompt=prompt,
                response=response,
                model=model,
                tokens=tokens
            )
    
    def _generate_cache_key(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        model: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate cache key from request parameters."""
        key_data = {
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()
        
        return f"inference:{key_hash}"
    
    async def _execute_inference(
        self,
        model_id: str,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        n: int,
        stop: Optional[List[str]],
        presence_penalty: float,
        frequency_penalty: float,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute model inference."""
        # Get model backend
        backend = await self.model_manager.get_backend(model_id)
        
        # Ensure model is loaded
        if not await self.model_manager.is_loaded(model_id):
            await self.model_manager.load_model(model_id)
        
        # Execute inference
        result = await backend.generate(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            timeout=timeout
        )
        
        return result
    
    async def _get_fallback_model(self, original_model: str) -> str:
        """Get fallback model for when primary model fails."""
        # Get model registry entry
        model = await self.model_registry.get_model(original_model)
        
        if model and model.get("fallback_models"):
            # Use configured fallback
            for fallback_id in model["fallback_models"]:
                fallback = await self.model_registry.get_model(fallback_id)
                if fallback and fallback.get("status") == "ready":
                    return fallback_id
        
        # Default fallback
        return self.config.default_fallback_model
    
    async def _log_request(
        self,
        request_id: str,
        prompt: str,
        response: str,
        model: str,
        tokens: int,
        latency_ms: float,
        cache_hit: bool,
        user_id: Optional[str] = None,
        cache_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log request to database."""
        log_entry = {
            "request_id": request_id,
            "prompt": prompt[:500],  # Truncate for storage
            "response": response[:1000],  # Truncate for storage
            "model": model,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "user_id": user_id,
            "timestamp": datetime.utcnow()
        }
        
        if cache_type:
            log_entry["cache_type"] = cache_type
        
        if metadata:
            log_entry["metadata"] = metadata
        
        await self.request_log_repo.create(log_entry)
    
    # ========================================================================
    # HEALTH & METRICS
    # ========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Get gateway health status."""
        return {
            "status": "healthy",
            "cache_enabled": self.config.enable_cache,
            "semantic_cache_enabled": self.config.enable_semantic_cache,
            "circuit_breakers_enabled": self.config.enable_circuit_breakers,
            "fallbacks_enabled": self.config.enable_fallbacks,
            "circuit_breakers": self.circuit_breaker_registry.get_summary(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get gateway statistics."""
        return {
            "config": {
                "max_prompt_length": self.config.max_prompt_length,
                "max_tokens": self.config.max_tokens,
                "cache_ttl": self.config.cache_ttl,
                "semantic_threshold": self.config.semantic_threshold,
                "request_timeout": self.config.request_timeout
            },
            "circuit_breakers": self.circuit_breaker_registry.get_summary()
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "GatewayHandler",
    "GatewayConfig"
]