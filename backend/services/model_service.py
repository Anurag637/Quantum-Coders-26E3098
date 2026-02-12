"""
Model Service - Production Ready
Orchestrates model lifecycle management, inference, and monitoring.
Provides high-level interface for model operations across different backends.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, AsyncGenerator
from datetime import datetime, timedelta
from uuid import uuid4

from core.logging import get_logger, get_model_logger
from core.exceptions import (
    ModelNotFoundError, ModelNotAvailableError, ModelInferenceError,
    ModelLoadingError, ModelUnloadingError, ModelTimeoutError,
    ModelMemoryError, QuantizationError
)
from config import settings
from models.model_manager import ModelManager
from models.model_registry import ModelRegistry
from models.model_loader import ModelLoader
from models.quantization_manager import QuantizationManager
from models.memory_manager import MemoryPool
from models.backends.base_backend import BaseBackend
from cache.cache_manager import CacheManager
from cache.semantic_cache import SemanticCache
from monitoring.metrics import MetricsCollector
from database.repositories.model_repository import ModelRepository
from utils.token_counter import TokenCounter
from utils.cost_calculator import CostCalculator

# Initialize loggers
logger = get_logger(__name__)
model_logger = get_model_logger()

# ============================================================================
# MODEL SERVICE
# ============================================================================

class ModelService:
    """
    High-level service for model operations.
    
    Features:
    - Model lifecycle management (load, unload, reload)
    - Inference with automatic fallback
    - Performance monitoring
    - Cost tracking
    - Cache integration
    - Batch processing
    """
    
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        model_registry: Optional[ModelRegistry] = None,
        model_loader: Optional[ModelLoader] = None,
        quantization_manager: Optional[QuantizationManager] = None,
        memory_pool: Optional[MemoryPool] = None,
        cache_manager: Optional[CacheManager] = None,
        semantic_cache: Optional[SemanticCache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        model_repository: Optional[ModelRepository] = None
    ):
        # Initialize components
        self.model_manager = model_manager or ModelManager()
        self.model_registry = model_registry or ModelRegistry()
        self.model_loader = model_loader or ModelLoader()
        self.quantization_manager = quantization_manager or QuantizationManager()
        self.memory_pool = memory_pool or MemoryPool()
        self.cache_manager = cache_manager or CacheManager()
        self.semantic_cache = semantic_cache or SemanticCache()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.model_repository = model_repository or ModelRepository()
        
        # Utilities
        self.token_counter = TokenCounter()
        self.cost_calculator = CostCalculator()
        
        # Performance tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            "model_service_initialized",
            model_manager_ready=bool(model_manager),
            cache_ready=bool(cache_manager),
            metrics_ready=bool(metrics_collector)
        )
    
    # ========================================================================
    # MODEL LIFECYCLE MANAGEMENT
    # ========================================================================
    
    async def load_model(
        self,
        model_id: str,
        quantization: Optional[str] = None,
        device: str = "auto",
        gpu_layers: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        force: bool = False,
        wait: bool = True,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load a model into memory.
        
        Args:
            model_id: Model identifier
            quantization: Quantization level (4bit, 8bit, fp16, bf16)
            device: Device to load on (cpu, cuda, mps, auto)
            gpu_layers: Number of layers to offload to GPU (llama.cpp)
            batch_size: Batch size for inference
            max_tokens: Maximum tokens to generate
            temperature: Default temperature
            force: Force load even if insufficient memory
            wait: Wait for loading to complete
            timeout: Loading timeout in seconds
        
        Returns:
            Dictionary with loading status
        
        Raises:
            ModelNotFoundError: Model not found in registry
            ModelMemoryError: Insufficient memory
            ModelLoadingError: Failed to load model
        """
        start_time = time.time()
        request_id = str(uuid4())
        
        logger.info(
            "model_load_requested",
            request_id=request_id,
            model_id=model_id,
            quantization=quantization,
            device=device,
            force=force
        )
        
        # Check if model exists
        model_config = await self.model_registry.get_model(model_id)
        if not model_config:
            raise ModelNotFoundError(
                model_id=model_id,
                detail=f"Model '{model_id}' not found in registry"
            )
        
        # Check if already loaded
        if await self.model_manager.is_loaded(model_id):
            logger.info(
                "model_already_loaded",
                request_id=request_id,
                model_id=model_id
            )
            
            return {
                "status": "already_loaded",
                "model_id": model_id,
                "loaded_at": await self.model_manager.get_load_time(model_id),
                "load_time_ms": 0
            }
        
        # Check memory requirements
        memory_required = model_config.get("memory_required_gb", 0)
        if memory_required > 0 and not force:
            has_memory = await self.memory_pool.allocate(
                model_id=model_id,
                memory_gb=memory_required,
                device=device
            )
            
            if not has_memory:
                available = await self.memory_pool.get_available_memory(device)
                raise ModelMemoryError(
                    model_id=model_id,
                    required_mb=memory_required * 1024,
                    available_mb=available * 1024,
                    detail=f"Insufficient memory to load model. Required: {memory_required}GB, Available: {available:.2f}GB"
                )
        
        # Load model
        try:
            load_task = asyncio.create_task(
                self.model_loader.load_model(
                    model_id=model_id,
                    quantization=quantization,
                    device=device,
                    gpu_layers=gpu_layers,
                    batch_size=batch_size,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            )
            
            if wait:
                if timeout:
                    success = await asyncio.wait_for(load_task, timeout=timeout)
                else:
                    success = await load_task
                
                load_time_ms = (time.time() - start_time) * 1000
                
                if success:
                    # Update model registry
                    await self.model_registry.update_model_status(
                        model_id=model_id,
                        status="ready",
                        is_loaded=True,
                        loaded_at=datetime.utcnow(),
                        metadata={
                            "quantization": quantization,
                            "device": device,
                            "load_time_ms": load_time_ms
                        }
                    )
                    
                    # Record metrics
                    await self.metrics_collector.record_model_load(
                        model_id=model_id,
                        success=True,
                        load_time_ms=load_time_ms,
                        memory_mb=memory_required * 1024
                    )
                    
                    model_logger.log_model_load(
                        model_id=model_id,
                        status="success",
                        duration_ms=load_time_ms,
                        memory_mb=memory_required * 1024
                    )
                    
                    logger.info(
                        "model_loaded",
                        request_id=request_id,
                        model_id=model_id,
                        load_time_ms=round(load_time_ms, 2),
                        quantization=quantization or model_config.get("quantization"),
                        device=device
                    )
                    
                    return {
                        "status": "loaded",
                        "model_id": model_id,
                        "load_time_ms": round(load_time_ms, 2),
                        "quantization": quantization or model_config.get("quantization"),
                        "device": device,
                        "memory_required_gb": memory_required
                    }
                else:
                    # Free allocated memory
                    if memory_required > 0:
                        await self.memory_pool.deallocate(model_id)
                    
                    raise ModelLoadingError(
                        model_id=model_id,
                        reason="Unknown error during model loading"
                    )
            else:
                # Background loading
                asyncio.create_task(self._background_load_task(
                    request_id=request_id,
                    model_id=model_id,
                    model_config=model_config,
                    quantization=quantization,
                    device=device,
                    gpu_layers=gpu_layers,
                    batch_size=batch_size,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    memory_required=memory_required,
                    load_task=load_task
                ))
                
                return {
                    "status": "loading",
                    "model_id": model_id,
                    "message": f"Model '{model_id}' is loading in background",
                    "request_id": request_id
                }
                
        except asyncio.TimeoutError:
            # Free allocated memory
            if memory_required > 0:
                await self.memory_pool.deallocate(model_id)
            
            raise ModelTimeoutError(
                model_id=model_id,
                timeout_seconds=timeout or 300,
                detail=f"Model loading timed out after {timeout} seconds"
            )
            
        except Exception as e:
            # Free allocated memory
            if memory_required > 0:
                await self.memory_pool.deallocate(model_id)
            
            logger.error(
                "model_load_failed",
                request_id=request_id,
                model_id=model_id,
                error=str(e),
                exc_info=True
            )
            
            await self.model_registry.update_model_status(
                model_id=model_id,
                status="error",
                error=str(e)
            )
            
            await self.metrics_collector.record_model_load(
                model_id=model_id,
                success=False,
                error=str(e)
            )
            
            model_logger.log_model_load(
                model_id=model_id,
                status="error",
                error=str(e)
            )
            
            raise ModelLoadingError(
                model_id=model_id,
                reason=str(e)
            )
    
    async def _background_load_task(
        self,
        request_id: str,
        model_id: str,
        model_config: Dict[str, Any],
        quantization: Optional[str],
        device: str,
        gpu_layers: Optional[int],
        batch_size: Optional[int],
        max_tokens: Optional[int],
        temperature: Optional[float],
        memory_required: float,
        load_task: asyncio.Task
    ):
        """Background task for model loading."""
        try:
            success = await load_task
            
            if success:
                load_time = (await self.model_manager.get_load_time(model_id)) or 0
                
                await self.model_registry.update_model_status(
                    model_id=model_id,
                    status="ready",
                    is_loaded=True,
                    loaded_at=datetime.utcnow(),
                    metadata={
                        "quantization": quantization,
                        "device": device,
                        "load_time_ms": load_time
                    }
                )
                
                await self.metrics_collector.record_model_load(
                    model_id=model_id,
                    success=True,
                    load_time_ms=load_time,
                    memory_mb=memory_required * 1024
                )
                
                model_logger.log_model_load(
                    model_id=model_id,
                    status="success",
                    duration_ms=load_time,
                    memory_mb=memory_required * 1024
                )
                
                logger.info(
                    "model_background_loaded",
                    request_id=request_id,
                    model_id=model_id,
                    load_time_ms=round(load_time, 2)
                )
            else:
                await self.memory_pool.deallocate(model_id)
                
                await self.model_registry.update_model_status(
                    model_id=model_id,
                    status="error",
                    error="Unknown loading error"
                )
                
                logger.error(
                    "model_background_load_failed",
                    request_id=request_id,
                    model_id=model_id
                )
                
        except Exception as e:
            await self.memory_pool.deallocate(model_id)
            
            await self.model_registry.update_model_status(
                model_id=model_id,
                status="error",
                error=str(e)
            )
            
            logger.error(
                "model_background_load_exception",
                request_id=request_id,
                model_id=model_id,
                error=str(e),
                exc_info=True
            )
    
    async def unload_model(
        self,
        model_id: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Unload a model from memory.
        
        Args:
            model_id: Model identifier
            force: Force unload even if in use
        
        Returns:
            Dictionary with unload status
        
        Raises:
            ModelNotFoundError: Model not found
            ModelUnloadingError: Failed to unload model
        """
        request_id = str(uuid4())
        
        logger.info(
            "model_unload_requested",
            request_id=request_id,
            model_id=model_id,
            force=force
        )
        
        # Check if model exists
        model_config = await self.model_registry.get_model(model_id)
        if not model_config:
            raise ModelNotFoundError(
                model_id=model_id,
                detail=f"Model '{model_id}' not found in registry"
            )
        
        # Check if loaded
        if not await self.model_manager.is_loaded(model_id):
            return {
                "status": "not_loaded",
                "model_id": model_id,
                "message": f"Model '{model_id}' is not loaded"
            }
        
        # Check active requests
        if not force:
            active = self.active_requests.get(model_id, [])
            if active:
                raise ModelUnloadingError(
                    model_id=model_id,
                    reason=f"Model has {len(active)} active requests. Use force=True to override."
                )
        
        try:
            start_time = time.time()
            
            # Unload model
            success = await self.model_loader.unload_model(
                model_id=model_id,
                force=force
            )
            
            if success:
                # Free memory
                await self.memory_pool.deallocate(model_id)
                
                # Update registry
                await self.model_registry.update_model_status(
                    model_id=model_id,
                    status="unloaded",
                    is_loaded=False,
                    loaded_at=None,
                    memory_usage_mb=0
                )
                
                unload_time_ms = (time.time() - start_time) * 1000
                
                # Record metrics
                await self.metrics_collector.record_model_unload(
                    model_id=model_id,
                    success=True,
                    unload_time_ms=unload_time_ms
                )
                
                model_logger.log_model_load(
                    model_id=model_id,
                    status="unloaded",
                    duration_ms=unload_time_ms
                )
                
                logger.info(
                    "model_unloaded",
                    request_id=request_id,
                    model_id=model_id,
                    unload_time_ms=round(unload_time_ms, 2)
                )
                
                return {
                    "status": "unloaded",
                    "model_id": model_id,
                    "unload_time_ms": round(unload_time_ms, 2)
                }
            else:
                raise ModelUnloadingError(
                    model_id=model_id,
                    reason="Unknown error during model unloading"
                )
                
        except Exception as e:
            logger.error(
                "model_unload_failed",
                request_id=request_id,
                model_id=model_id,
                error=str(e),
                exc_info=True
            )
            
            await self.metrics_collector.record_model_unload(
                model_id=model_id,
                success=False,
                error=str(e)
            )
            
            raise ModelUnloadingError(
                model_id=model_id,
                reason=str(e)
            )
    
    async def reload_model(
        self,
        model_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Reload a model (unload + load).
        
        Args:
            model_id: Model identifier
            **kwargs: Loading parameters
        
        Returns:
            Dictionary with reload status
        """
        await self.unload_model(model_id, force=True)
        return await self.load_model(model_id, **kwargs)
    
    async def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed model status.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Dictionary with model status
        """
        # Get model configuration
        config = await self.model_registry.get_model(model_id)
        if not config:
            raise ModelNotFoundError(model_id=model_id)
        
        # Get runtime status
        is_loaded = await self.model_manager.is_loaded(model_id)
        load_time = await self.model_manager.get_load_time(model_id) if is_loaded else None
        memory_usage = await self.model_manager.get_model_memory_usage(model_id) if is_loaded else None
        
        # Get performance metrics
        metrics = await self.metrics_collector.get_model_metrics(
            model_id=model_id,
            hours=24
        )
        
        # Get active requests
        active_requests = self.active_requests.get(model_id, [])
        
        return {
            "model_id": model_id,
            "name": config.get("name", model_id),
            "provider": config.get("provider", "unknown"),
            "type": config.get("type", "unknown"),
            "status": config.get("status", "unknown"),
            "is_loaded": is_loaded,
            "loaded_at": load_time.isoformat() if load_time else None,
            "memory_usage_mb": round(memory_usage, 2) if memory_usage else None,
            "active_requests": len(active_requests),
            "capabilities": config.get("capabilities", []),
            "quantization": config.get("quantization"),
            "context_size": config.get("context_size", 2048),
            "performance": {
                "total_requests": metrics.get("total_requests", 0),
                "success_rate": round(metrics.get("success_rate", 100), 2),
                "avg_latency_ms": round(metrics.get("avg_latency_ms", 0), 2),
                "p95_latency_ms": round(metrics.get("p95_latency_ms", 0), 2),
                "tokens_per_second": round(metrics.get("tokens_per_second", 0), 2),
                "cost_per_request": round(metrics.get("cost_per_request", 0), 6)
            },
            "metadata": config.get("metadata", {})
        }
    
    # ========================================================================
    # INFERENCE
    # ========================================================================
    
    async def generate(
        self,
        model_id: str,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        n: int = 1,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stream: bool = False,
        use_cache: bool = True,
        timeout: Optional[int] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using the specified model.
        
        Args:
            model_id: Model identifier
            prompt: Input prompt
            messages: Chat messages (for chat models)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            n: Number of completions
            stop: Stop sequences
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            stream: Stream response
            use_cache: Use cache
            timeout: Request timeout
            request_id: Request ID for tracking
            user_id: User ID for tracking
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with generated text and metadata
        
        Raises:
            ModelNotAvailableError: Model not available
            ModelInferenceError: Inference failed
            ModelTimeoutError: Request timed out
        """
        request_id = request_id or str(uuid4())
        start_time = time.time()
        
        logger.info(
            "model_inference_requested",
            request_id=request_id,
            model_id=model_id,
            prompt_length=len(prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            user_id=user_id
        )
        
        # Check if model is loaded
        if not await self.model_manager.is_loaded(model_id):
            # Try to load model
            try:
                await self.load_model(model_id, wait=True, timeout=30)
            except Exception as e:
                raise ModelNotAvailableError(
                    model_id=model_id,
                    status="not_loaded",
                    detail=f"Model not loaded and failed to load: {str(e)}"
                )
        
        # Check cache if enabled
        cache_hit = False
        cached_response = None
        
        if use_cache and settings.cache.enabled:
            cache_key = f"inference:{model_id}:{hash(prompt)}:{max_tokens}:{temperature}"
            cached_response = await self.cache_manager.get(cache_key)
            
            if cached_response:
                cache_hit = True
                logger.debug(
                    "inference_cache_hit",
                    request_id=request_id,
                    model_id=model_id,
                    cache_key=cache_key
                )
                
                # Record cache hit
                await self.metrics_collector.record_cache_hit(
                    cache_type="exact",
                    model_id=model_id
                )
                
                # Calculate tokens saved
                response_text = cached_response["content"]
                tokens_generated = self.token_counter.count_tokens(response_text)
                tokens_saved = tokens_generated
                cost_saved = self.cost_calculator.calculate_cost(
                    model=model_id,
                    completion_tokens=tokens_saved
                )
                
                return {
                    "id": f"cmpl-{request_id[:8]}",
                    "content": response_text,
                    "model": model_id,
                    "tokens": tokens_generated,
                    "finish_reason": "stop",
                    "cache_hit": True,
                    "cache_type": "exact",
                    "latency_ms": (time.time() - start_time) * 1000,
                    "tokens_saved": tokens_saved,
                    "cost_saved_usd": round(cost_saved, 6)
                }
        
        # Track active request
        if model_id not in self.active_requests:
            self.active_requests[model_id] = []
        self.active_requests[model_id].append(request_id)
        
        try:
            # Get model backend
            backend = await self.model_manager.get_backend(model_id)
            
            # Execute inference
            if stream:
                # Streaming response
                return await self._generate_stream(
                    backend=backend,
                    model_id=model_id,
                    prompt=prompt,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    request_id=request_id,
                    start_time=start_time,
                    user_id=user_id
                )
            else:
                # Non-streaming response
                if timeout:
                    result = await asyncio.wait_for(
                        backend.generate(
                            prompt=prompt,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            n=n,
                            stop=stop,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            **kwargs
                        ),
                        timeout=timeout
                    )
                else:
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
                        **kwargs
                    )
                
                inference_time = time.time() - start_time
                latency_ms = inference_time * 1000
                
                # Get response content
                if isinstance(result, dict):
                    content = result.get("content", "")
                    finish_reason = result.get("finish_reason", "stop")
                else:
                    content = result
                    finish_reason = "stop"
                
                # Calculate token usage
                prompt_tokens = self.token_counter.count_tokens(prompt)
                completion_tokens = self.token_counter.count_tokens(content)
                total_tokens = prompt_tokens + completion_tokens
                
                # Calculate cost
                cost = self.cost_calculator.calculate_cost(
                    model=model_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                
                # Cache response if enabled
                if use_cache and settings.cache.enabled and not cache_hit:
                    cache_key = f"inference:{model_id}:{hash(prompt)}:{max_tokens}:{temperature}"
                    asyncio.create_task(
                        self.cache_manager.set(
                            cache_key,
                            {
                                "content": content,
                                "model": model_id,
                                "tokens": total_tokens
                            },
                            ttl=settings.cache.default_ttl
                        )
                    )
                
                # Record metrics
                await self.metrics_collector.record_inference(
                    model_id=model_id,
                    latency_ms=latency_ms,
                    tokens=total_tokens,
                    cost=cost,
                    success=True,
                    user_id=user_id
                )
                
                model_logger.log_model_inference(
                    model_id=model_id,
                    request_id=request_id,
                    latency_ms=latency_ms,
                    tokens=total_tokens,
                    success=True,
                    cache_hit=cache_hit,
                    cost=cost
                )
                
                logger.info(
                    "model_inference_completed",
                    request_id=request_id,
                    model_id=model_id,
                    latency_ms=round(latency_ms, 2),
                    tokens=total_tokens,
                    cache_hit=cache_hit
                )
                
                return {
                    "id": f"cmpl-{request_id[:8]}",
                    "content": content,
                    "model": model_id,
                    "tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                    "cache_hit": cache_hit,
                    "latency_ms": round(latency_ms, 2),
                    "cost_usd": round(cost, 6),
                    "tokens_per_second": round(completion_tokens / inference_time, 2) if inference_time > 0 else 0
                }
                
        except asyncio.TimeoutError:
            raise ModelTimeoutError(
                model_id=model_id,
                timeout_seconds=timeout or 60,
                detail=f"Inference timed out after {timeout} seconds"
            )
            
        except Exception as e:
            logger.error(
                "model_inference_failed",
                request_id=request_id,
                model_id=model_id,
                error=str(e),
                exc_info=True
            )
            
            await self.metrics_collector.record_inference(
                model_id=model_id,
                latency_ms=(time.time() - start_time) * 1000,
                tokens=0,
                cost=0,
                success=False,
                error=str(e),
                user_id=user_id
            )
            
            model_logger.log_model_inference(
                model_id=model_id,
                request_id=request_id,
                latency_ms=(time.time() - start_time) * 1000,
                tokens=0,
                success=False,
                error=str(e)
            )
            
            raise ModelInferenceError(
                model_id=model_id,
                reason=str(e)
            )
            
        finally:
            # Remove active request
            if model_id in self.active_requests:
                try:
                    self.active_requests[model_id].remove(request_id)
                except ValueError:
                    pass
    
    async def _generate_stream(
        self,
        backend: BaseBackend,
        model_id: str,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        presence_penalty: float,
        frequency_penalty: float,
        request_id: str,
        start_time: float,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response."""
        full_response = ""
        token_count = 0
        
        try:
            async for chunk in backend.generate_stream(
                prompt=prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            ):
                full_response += chunk
                token_count += 1
                
                yield {
                    "id": f"cmpl-{request_id[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
            
            # Final chunk
            yield {
                "id": f"cmpl-{request_id[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            
            # Calculate metrics
            inference_time = time.time() - start_time
            latency_ms = inference_time * 1000
            
            prompt_tokens = self.token_counter.count_tokens(prompt)
            completion_tokens = self.token_counter.count_tokens(full_response)
            total_tokens = prompt_tokens + completion_tokens
            
            cost = self.cost_calculator.calculate_cost(
                model=model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
            # Record metrics
            await self.metrics_collector.record_inference(
                model_id=model_id,
                latency_ms=latency_ms,
                tokens=total_tokens,
                cost=cost,
                success=True,
                streaming=True,
                user_id=user_id
            )
            
            model_logger.log_model_inference(
                model_id=model_id,
                request_id=request_id,
                latency_ms=latency_ms,
                tokens=total_tokens,
                success=True,
                streaming=True,
                cost=cost
            )
            
            logger.info(
                "model_streaming_completed",
                request_id=request_id,
                model_id=model_id,
                latency_ms=round(latency_ms, 2),
                tokens=total_tokens,
                tokens_per_second=round(token_count / inference_time, 2)
            )
            
        except Exception as e:
            logger.error(
                "model_streaming_failed",
                request_id=request_id,
                model_id=model_id,
                error=str(e),
                exc_info=True
            )
            
            await self.metrics_collector.record_inference(
                model_id=model_id,
                latency_ms=(time.time() - start_time) * 1000,
                tokens=0,
                cost=0,
                success=False,
                error=str(e),
                streaming=True,
                user_id=user_id
            )
            
            raise
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    async def generate_batch(
        self,
        model_id: str,
        prompts: List[str],
        max_concurrent: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            model_id: Model identifier
            prompts: List of prompts
            max_concurrent: Maximum concurrent requests
            **kwargs: Generation parameters
        
        Returns:
            List of generation results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _generate_with_semaphore(prompt: str, index: int):
            async with semaphore:
                try:
                    result = await self.generate(
                        model_id=model_id,
                        prompt=prompt,
                        **kwargs
                    )
                    result["index"] = index
                    return result
                except Exception as e:
                    return {
                        "index": index,
                        "prompt": prompt,
                        "error": str(e),
                        "success": False
                    }
        
        tasks = [
            _generate_with_semaphore(prompt, i)
            for i, prompt in enumerate(prompts)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Sort by original index
        results.sort(key=lambda x: x["index"])
        
        return results
    
    # ========================================================================
    # MODEL INFORMATION
    # ========================================================================
    
    async def list_models(
        self,
        status: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available models with optional filtering.
        
        Args:
            status: Filter by status
            capabilities: Filter by capabilities
            provider: Filter by provider
        
        Returns:
            List of model configurations
        """
        models = await self.model_registry.get_all_models()
        
        # Apply filters
        filtered = []
        for model in models:
            if status and model.get("status") != status:
                continue
            
            if provider and model.get("provider") != provider:
                continue
            
            if capabilities:
                model_caps = model.get("capabilities", [])
                if not all(cap in model_caps for cap in capabilities):
                    continue
            
            # Add runtime info
            model_id = model["id"]
            model["is_loaded"] = await self.model_manager.is_loaded(model_id)
            
            if model["is_loaded"]:
                model["memory_usage_mb"] = await self.model_manager.get_model_memory_usage(model_id)
            
            filtered.append(model)
        
        return filtered
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Dictionary with model information
        """
        return await self.get_model_status(model_id)
    
    # ========================================================================
    # QUANTIZATION
    # ========================================================================
    
    async def quantize_model(
        self,
        model_id: str,
        quantization: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Quantize a model to reduce memory usage.
        
        Args:
            model_id: Model identifier
            quantization: Quantization level (4bit, 8bit, fp16)
            output_path: Output path for quantized model
        
        Returns:
            Dictionary with quantization status
        
        Raises:
            ModelNotFoundError: Model not found
            QuantizationError: Quantization failed
        """
        request_id = str(uuid4())
        
        logger.info(
            "model_quantization_requested",
            request_id=request_id,
            model_id=model_id,
            quantization=quantization
        )
        
        # Check if model exists
        model_config = await self.model_registry.get_model(model_id)
        if not model_config:
            raise ModelNotFoundError(
                model_id=model_id,
                detail=f"Model '{model_id}' not found in registry"
            )
        
        # Check if model can be quantized
        if model_config.get("type") == "external":
            raise QuantizationError(
                model_id=model_id,
                quantization=quantization,
                reason="External API models cannot be quantized"
            )
        
        try:
            start_time = time.time()
            
            # Quantize model
            success = await self.quantization_manager.quantize_model(
                model_id=model_id,
                quantization=quantization,
                output_path=output_path
            )
            
            if success:
                quantize_time_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    "model_quantized",
                    request_id=request_id,
                    model_id=model_id,
                    quantization=quantization,
                    quantize_time_ms=round(quantize_time_ms, 2)
                )
                
                return {
                    "status": "quantized",
                    "model_id": model_id,
                    "quantization": quantization,
                    "quantize_time_ms": round(quantize_time_ms, 2),
                    "output_path": output_path or f"models/{model_id}-{quantization}"
                }
            else:
                raise QuantizationError(
                    model_id=model_id,
                    quantization=quantization,
                    reason="Unknown error during quantization"
                )
                
        except Exception as e:
            logger.error(
                "model_quantization_failed",
                request_id=request_id,
                model_id=model_id,
                quantization=quantization,
                error=str(e),
                exc_info=True
            )
            
            raise QuantizationError(
                model_id=model_id,
                quantization=quantization,
                reason=str(e)
            )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_model_service = None


def get_model_service() -> ModelService:
    """Get singleton model service instance."""
    global _model_service
    if not _model_service:
        _model_service = ModelService()
    return _model_service


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ModelService",
    "get_model_service"
]