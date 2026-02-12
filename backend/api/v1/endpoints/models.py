"""
Model Management Endpoints - Production Ready
Complete CRUD operations for LLM models with loading, unloading, and monitoring
Supports 15+ models across multiple backends (HuggingFace, vLLM, llama.cpp, external APIs)
"""

from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List, Optional, Union
import time
import uuid
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field, validator

from core.logging import get_logger
from core.security import verify_api_key, verify_admin, get_current_user
from core.exceptions import ModelNotAvailableError, ValidationError
from core.rate_limiter import rate_limit
from config import settings
from models.model_manager import ModelManager
from models.model_registry import ModelRegistry
from models.model_loader import ModelLoader
from models.quantization_manager import QuantizationManager
from models.backends.base_backend import BaseBackend
from models.backends.huggingface_backend import HuggingFaceBackend
from models.backends.grok_backend import GrokBackend
from models.backends.llamacpp_backend import LlamaCppBackend
from models.backends.vllm_backend import VLLMBackend
from models.backends.openai_backend import OpenAIBackend
from models.backends.anthropic_backend import AnthropicBackend
from models.backends.cohere_backend import CohereBackend
from database.repositories.model_repository import ModelRepository
from monitoring.metrics import MetricsCollector

# Initialize router
router = APIRouter(prefix="/models", tags=["Models"])

# Initialize logger
logger = get_logger(__name__)

# Initialize services
model_manager = ModelManager()
model_registry = ModelRegistry()
model_loader = ModelLoader()
quantization_manager = QuantizationManager()
model_repository = ModelRepository()
metrics_collector = MetricsCollector()

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class ModelInfo(BaseModel):
    """Model information schema"""
    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    provider: str = Field(..., description="Model provider (Meta, Mistral, OpenAI, etc.)")
    type: str = Field(..., description="Model type (llama, mistral, gpt, external, etc.)")
    library: str = Field(..., description="Backend library (transformers, llama-cpp, vllm, etc.)")
    format: Optional[str] = Field(None, description="Model format (gguf, safetensors, api)")
    quantization: Optional[str] = Field(None, description="Quantization level (4bit, 8bit, fp16)")
    context_size: int = Field(..., description="Maximum context window size")
    capabilities: List[str] = Field(..., description="Model capabilities (chat, code, reasoning, etc.)")
    status: str = Field(..., description="Current status (available, loading, ready, error, unloaded)")
    is_loaded: bool = Field(False, description="Whether model is currently loaded in memory")
    loaded_at: Optional[datetime] = Field(None, description="When model was loaded")
    memory_usage_mb: Optional[float] = Field(None, description="Current memory usage in MB")
    requires_api_key: bool = Field(False, description="Whether model requires API key")
    api_key_configured: Optional[bool] = Field(None, description="Whether API key is configured")
    cost_per_token: Optional[float] = Field(None, description="Cost per token in USD")
    latency_p95_ms: Optional[float] = Field(None, description="95th percentile latency in ms")
    error_count: int = Field(0, description="Number of errors encountered")
    total_requests: int = Field(0, description="Total number of requests")
    created_at: datetime = Field(..., description="When model was added to registry")
    updated_at: datetime = Field(..., description="When model was last updated")
    
    class Config:
        from_attributes = True


class ModelDetailInfo(ModelInfo):
    """Detailed model information including configuration"""
    config: Dict[str, Any] = Field(..., description="Full model configuration")
    download_url: Optional[str] = Field(None, description="URL to download model")
    file_size_mb: Optional[float] = Field(None, description="Model file size in MB")
    memory_required_gb: Optional[float] = Field(None, description="Minimum memory required in GB")
    gpu_memory_required_gb: Optional[float] = Field(None, description="Minimum GPU memory required in GB")
    recommended_batch_size: Optional[int] = Field(None, description="Recommended batch size")
    quantization_options: List[Dict[str, Any]] = Field(default_factory=list, description="Available quantization options")
    fallback_models: List[str] = Field(default_factory=list, description="Fallback model IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ModelLoadRequest(BaseModel):
    """Request schema for loading a model"""
    model_id: str = Field(..., description="Model ID to load")
    quantization: Optional[str] = Field(None, description="Quantization level (4bit, 8bit, fp16, bf16)")
    device: Optional[str] = Field("auto", description="Device to load on (cpu, cuda, auto)")
    gpu_layers: Optional[int] = Field(None, description="Number of layers to offload to GPU (for llama.cpp)")
    batch_size: Optional[int] = Field(None, description="Batch size for inference")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Default temperature")
    
    @validator('quantization')
    def validate_quantization(cls, v):
        if v and v not in ['4bit', '8bit', 'fp16', 'bf16', 'fp32']:
            raise ValueError('Quantization must be 4bit, 8bit, fp16, bf16, or fp32')
        return v
    
    @validator('device')
    def validate_device(cls, v):
        if v not in ['auto', 'cpu', 'cuda', 'mps']:
            raise ValueError('Device must be auto, cpu, cuda, or mps')
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v and (v < 0 or v > 2):
            raise ValueError('Temperature must be between 0 and 2')
        return v


class ModelUnloadRequest(BaseModel):
    """Request schema for unloading a model"""
    model_id: str = Field(..., description="Model ID to unload")
    force: bool = Field(False, description="Force unload even if in use")


class ModelUpdateRequest(BaseModel):
    """Request schema for updating model configuration"""
    model_id: str = Field(..., description="Model ID to update")
    name: Optional[str] = Field(None, description="Updated model name")
    capabilities: Optional[List[str]] = Field(None, description="Updated capabilities")
    fallback_models: Optional[List[str]] = Field(None, description="Updated fallback models")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    is_active: Optional[bool] = Field(None, description="Whether model is active")


class ModelTestRequest(BaseModel):
    """Request schema for testing a model"""
    model_id: str = Field(..., description="Model ID to test")
    prompt: str = Field(..., description="Test prompt", min_length=1, max_length=1000)
    max_tokens: int = Field(50, description="Maximum tokens to generate", ge=1, le=500)
    temperature: float = Field(0.7, description="Temperature", ge=0, le=2)


class ModelTestResponse(BaseModel):
    """Response schema for model test"""
    model_id: str = Field(..., description="Model ID tested")
    prompt: str = Field(..., description="Test prompt")
    response: str = Field(..., description="Model response")
    latency_ms: float = Field(..., description="Latency in milliseconds")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    tokens_per_second: float = Field(..., description="Tokens per second")
    success: bool = Field(..., description="Whether test was successful")
    error: Optional[str] = Field(None, description="Error message if failed")


class ModelMetricsResponse(BaseModel):
    """Response schema for model metrics"""
    model_id: str = Field(..., description="Model ID")
    timestamp: datetime = Field(..., description="Metrics timestamp")
    request_count: int = Field(..., description="Total requests")
    success_count: int = Field(..., description="Successful requests")
    error_count: int = Field(..., description="Failed requests")
    avg_latency_ms: float = Field(..., description="Average latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    p99_latency_ms: float = Field(..., description="99th percentile latency")
    total_tokens: int = Field(..., description="Total tokens generated")
    avg_tokens_per_request: float = Field(..., description="Average tokens per request")
    tokens_per_second: float = Field(..., description="Tokens per second")
    memory_usage_mb: Optional[float] = Field(None, description="Current memory usage")
    gpu_usage_percent: Optional[float] = Field(None, description="GPU utilization")
    cost_total: float = Field(0, description="Total cost in USD")
    cost_per_request: float = Field(0, description="Average cost per request")


class ModelComparisonRequest(BaseModel):
    """Request schema for comparing models"""
    model_ids: List[str] = Field(..., description="Model IDs to compare", min_items=2, max_items=5)
    prompt: str = Field(..., description="Prompt to test with", min_length=1, max_length=1000)
    max_tokens: int = Field(100, description="Maximum tokens to generate", ge=1, le=500)


class ModelComparisonResponse(BaseModel):
    """Response schema for model comparison"""
    prompt: str = Field(..., description="Test prompt")
    timestamp: datetime = Field(..., description="Comparison timestamp")
    results: List[Dict[str, Any]] = Field(..., description="Results for each model")
    recommendations: List[str] = Field(default_factory=list, description="Model recommendations")


# ============================================================================
# PUBLIC ENDPOINTS - LIST AND GET MODELS
# ============================================================================

@router.get(
    "",
    summary="List Models",
    description="""
    List all available models in the registry.
    
    Returns all configured models with their current status, capabilities, and metadata.
    Supports filtering by status, type, capability, and provider.
    
    Features:
    - Pagination for large model lists
    - Filtering by multiple criteria
    - Sorting by various fields
    - Cached responses (60s TTL)
    - Rate limited: 30 requests/minute
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def list_models(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status (available, loading, ready, error, unloaded)"),
    model_type: Optional[str] = Query(None, description="Filter by model type (llama, mistral, gpt, external)"),
    capability: Optional[str] = Query(None, description="Filter by capability (chat, code, reasoning, vision)"),
    provider: Optional[str] = Query(None, description="Filter by provider (Meta, Mistral, OpenAI, Anthropic)"),
    is_loaded: Optional[bool] = Query(None, description="Filter by load status"),
    search: Optional[str] = Query(None, description="Search in model name and ID"),
    sort_by: str = Query("name", description="Sort field (name, id, provider, created_at)"),
    sort_order: str = Query("asc", description="Sort order (asc, desc)"),
    limit: int = Query(50, description="Number of models to return", ge=1, le=100),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    List all available models with filtering and pagination.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "list_models_requested",
        request_id=request_id,
        status=status,
        model_type=model_type,
        capability=capability,
        limit=limit,
        offset=offset
    )
    
    try:
        # Get all models from registry
        all_models = await model_registry.get_all_models()
        
        # Apply filters
        filtered_models = []
        for model in all_models:
            # Status filter
            if status and model.get("status") != status:
                continue
            
            # Type filter
            if model_type and model.get("type") != model_type:
                continue
            
            # Provider filter
            if provider and model.get("provider", "").lower() != provider.lower():
                continue
            
            # Capability filter
            if capability:
                capabilities = model.get("capabilities", [])
                if capability not in capabilities:
                    continue
            
            # Load status filter
            if is_loaded is not None and model.get("is_loaded") != is_loaded:
                continue
            
            # Search filter
            if search:
                search_lower = search.lower()
                model_id = model.get("id", "").lower()
                model_name = model.get("name", "").lower()
                if search_lower not in model_id and search_lower not in model_name:
                    continue
            
            filtered_models.append(model)
        
        # Sort models
        reverse = sort_order.lower() == "desc"
        if sort_by == "name":
            filtered_models.sort(key=lambda x: x.get("name", ""), reverse=reverse)
        elif sort_by == "id":
            filtered_models.sort(key=lambda x: x.get("id", ""), reverse=reverse)
        elif sort_by == "provider":
            filtered_models.sort(key=lambda x: x.get("provider", ""), reverse=reverse)
        elif sort_by == "created_at":
            filtered_models.sort(key=lambda x: x.get("created_at", 0), reverse=reverse)
        
        # Apply pagination
        total = len(filtered_models)
        paginated_models = filtered_models[offset:offset + limit]
        
        # Convert to response models
        models_response = []
        for model in paginated_models:
            # Check if API key is configured for external models
            api_key_configured = None
            if model.get("requires_api_key"):
                provider_key = f"{model.get('provider', '').lower()}_api_key"
                if provider_key in ["grok_api_key", "openai_api_key", "anthropic_api_key", "cohere_api_key"]:
                    api_key_configured = getattr(settings.api, provider_key) is not None
            
            models_response.append(ModelInfo(
                id=model["id"],
                name=model.get("name", model["id"]),
                provider=model.get("provider", "Unknown"),
                type=model.get("type", "unknown"),
                library=model.get("library", "unknown"),
                format=model.get("format"),
                quantization=model.get("quantization"),
                context_size=model.get("context_size", 2048),
                capabilities=model.get("capabilities", []),
                status=model.get("status", "unknown"),
                is_loaded=model.get("is_loaded", False),
                loaded_at=model.get("loaded_at"),
                memory_usage_mb=model.get("memory_usage_mb"),
                requires_api_key=model.get("requires_api_key", False),
                api_key_configured=api_key_configured,
                cost_per_token=model.get("cost_per_token"),
                latency_p95_ms=model.get("latency_p95_ms"),
                error_count=model.get("error_count", 0),
                total_requests=model.get("total_requests", 0),
                created_at=model.get("created_at", datetime.now()),
                updated_at=model.get("updated_at", datetime.now())
            ))
        
        response = {
            "object": "list",
            "data": models_response,
            "total": total,
            "limit": limit,
            "offset": offset,
            "request_id": request_id,
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        logger.info(
            "list_models_completed",
            request_id=request_id,
            total=total,
            returned=len(models_response),
            response_time_ms=response["response_time_ms"]
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "list_models_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to list models",
                "request_id": request_id
            }
        )


@router.get(
    "/{model_id}",
    summary="Get Model",
    description="""
    Get detailed information about a specific model.
    
    Returns complete model configuration, status, metrics, and metadata.
    Includes real-time load status and performance metrics.
    """,
    response_model=ModelDetailInfo,
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def get_model(
    request: Request,
    model_id: str,
    api_key: str = Depends(verify_api_key)
) -> ModelDetailInfo:
    """
    Get detailed information about a specific model.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "get_model_requested",
        request_id=request_id,
        model_id=model_id
    )
    
    try:
        # Get model from registry
        model = await model_registry.get_model(model_id)
        
        if not model:
            logger.warning(
                "model_not_found",
                request_id=request_id,
                model_id=model_id
            )
            
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ModelNotFound",
                    "message": f"Model '{model_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Get real-time status
        is_loaded = await model_manager.is_loaded(model_id)
        status = await model_manager.get_model_status(model_id)
        
        # Get performance metrics
        metrics = await metrics_collector.get_model_metrics(model_id, hours=24)
        
        # Check API key configuration for external models
        api_key_configured = None
        if model.get("requires_api_key"):
            provider = model.get("provider", "").lower()
            if "grok" in provider or "xai" in provider:
                api_key_configured = settings.grok_api_key_value is not None
            elif "openai" in provider:
                api_key_configured = settings.openai_api_key_value is not None
            elif "anthropic" in provider:
                api_key_configured = settings.anthropic_api_key_value is not None
            elif "cohere" in provider:
                api_key_configured = settings.cohere_api_key_value is not None
        
        response = ModelDetailInfo(
            id=model["id"],
            name=model.get("name", model["id"]),
            provider=model.get("provider", "Unknown"),
            type=model.get("type", "unknown"),
            library=model.get("library", "unknown"),
            format=model.get("format"),
            quantization=model.get("quantization"),
            context_size=model.get("context_size", 2048),
            capabilities=model.get("capabilities", []),
            status=status or model.get("status", "unknown"),
            is_loaded=is_loaded,
            loaded_at=model.get("loaded_at"),
            memory_usage_mb=model.get("memory_usage_mb"),
            requires_api_key=model.get("requires_api_key", False),
            api_key_configured=api_key_configured,
            cost_per_token=model.get("cost_per_token"),
            latency_p95_ms=metrics.get("p95_latency_ms") if metrics else None,
            error_count=model.get("error_count", 0),
            total_requests=model.get("total_requests", 0),
            created_at=model.get("created_at", datetime.now()),
            updated_at=model.get("updated_at", datetime.now()),
            config=model.get("config", {}),
            download_url=model.get("download_url"),
            file_size_mb=model.get("file_size_mb"),
            memory_required_gb=model.get("memory_required_gb"),
            gpu_memory_required_gb=model.get("gpu_memory_required_gb"),
            recommended_batch_size=model.get("recommended_batch_size"),
            quantization_options=model.get("quantization_options", []),
            fallback_models=model.get("fallback_models", []),
            metadata=model.get("metadata", {})
        )
        
        logger.info(
            "get_model_completed",
            request_id=request_id,
            model_id=model_id,
            response_time_ms=round((time.time() - start_time) * 1000, 2)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "get_model_failed",
            request_id=request_id,
            model_id=model_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to get model '{model_id}'",
                "request_id": request_id
            }
        )


# ============================================================================
# MODEL LOADING AND UNLOADING ENDPOINTS
# ============================================================================

@router.post(
    "/load",
    summary="Load Model",
    description="""
    Load a model into memory for inference.
    
    This endpoint:
    1. Validates model availability and requirements
    2. Allocates memory (GPU/CPU)
    3. Loads model using appropriate backend
    4. Performs warm-up inference
    5. Registers model as ready
    
    Supports:
    - Dynamic quantization selection
    - GPU layer offloading for llama.cpp
    - Custom device selection
    - Batch size configuration
    
    Rate limited: 10 requests per minute
    """,
    response_model=Dict[str, Any],
    status_code=202,
    dependencies=[Depends(rate_limit(limit=10, period=60))]
)
async def load_model(
    request: Request,
    load_request: ModelLoadRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Load a model into memory asynchronously.
    Returns immediately with task ID for tracking.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "load_model_requested",
        request_id=request_id,
        model_id=load_request.model_id,
        quantization=load_request.quantization,
        device=load_request.device
    )
    
    try:
        # Check if model exists
        model = await model_registry.get_model(load_request.model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ModelNotFound",
                    "message": f"Model '{load_request.model_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Check if model is already loaded
        if await model_manager.is_loaded(load_request.model_id):
            return {
                "status": "already_loaded",
                "model_id": load_request.model_id,
                "message": f"Model '{load_request.model_id}' is already loaded",
                "request_id": request_id
            }
        
        # Check API key for external models
        if model.get("requires_api_key"):
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
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "APIKeyRequired",
                        "message": f"API key required for {model.get('provider')} models",
                        "provider": model.get("provider"),
                        "request_id": request_id
                    }
                )
        
        # Check memory requirements
        memory_required = model.get("memory_required_gb", 0)
        if memory_required > 0:
            from models.memory_manager import MemoryPool
            memory_pool = MemoryPool()
            
            if not memory_pool.allocate(
                load_request.model_id,
                memory_required,
                device=load_request.device
            ):
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "InsufficientMemory",
                        "message": f"Insufficient memory to load model ({memory_required}GB required)",
                        "memory_required_gb": memory_required,
                        "request_id": request_id
                    }
                )
        
        # Generate task ID for tracking
        task_id = str(uuid.uuid4())
        
        # Add to background tasks
        background_tasks.add_task(
            _load_model_task,
            task_id=task_id,
            request_id=request_id,
            model_id=load_request.model_id,
            quantization=load_request.quantization,
            device=load_request.device,
            gpu_layers=load_request.gpu_layers,
            batch_size=load_request.batch_size,
            max_tokens=load_request.max_tokens,
            temperature=load_request.temperature
        )
        
        response = {
            "status": "loading",
            "task_id": task_id,
            "model_id": load_request.model_id,
            "message": f"Loading model '{load_request.model_id}' in background",
            "estimated_time_seconds": 30,  # Estimate
            "request_id": request_id,
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        logger.info(
            "load_model_accepted",
            request_id=request_id,
            model_id=load_request.model_id,
            task_id=task_id
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "load_model_failed",
            request_id=request_id,
            model_id=load_request.model_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to load model '{load_request.model_id}'",
                "request_id": request_id
            }
        )


async def _load_model_task(
    task_id: str,
    request_id: str,
    model_id: str,
    quantization: Optional[str],
    device: str,
    gpu_layers: Optional[int],
    batch_size: Optional[int],
    max_tokens: Optional[int],
    temperature: Optional[float]
):
    """
    Background task for loading models.
    """
    logger.info(
        "load_model_task_started",
        task_id=task_id,
        request_id=request_id,
        model_id=model_id
    )
    
    try:
        # Update model status
        await model_registry.update_model_status(model_id, "loading")
        
        # Load the model
        success = await model_loader.load_model(
            model_id=model_id,
            quantization=quantization,
            device=device,
            gpu_layers=gpu_layers,
            batch_size=batch_size,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if success:
            # Update model status
            await model_registry.update_model_status(
                model_id,
                "ready",
                is_loaded=True,
                loaded_at=datetime.now()
            )
            
            logger.info(
                "load_model_task_completed",
                task_id=task_id,
                request_id=request_id,
                model_id=model_id
            )
        else:
            # Update model status
            await model_registry.update_model_status(
                model_id,
                "error",
                error="Failed to load model"
            )
            
            logger.error(
                "load_model_task_failed",
                task_id=task_id,
                request_id=request_id,
                model_id=model_id,
                error="Failed to load model"
            )
            
    except Exception as e:
        # Update model status
        await model_registry.update_model_status(
            model_id,
            "error",
            error=str(e)
        )
        
        logger.error(
            "load_model_task_exception",
            task_id=task_id,
            request_id=request_id,
            model_id=model_id,
            error=str(e),
            exc_info=True
        )


@router.post(
    "/unload",
    summary="Unload Model",
    description="""
    Unload a model from memory to free resources.
    
    This endpoint:
    1. Checks if model is currently loaded
    2. Stops any ongoing inference
    3. Frees GPU/CPU memory
    4. Updates model status
    
    Supports force unload for stuck models.
    
    Rate limited: 10 requests per minute
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(rate_limit(limit=10, period=60))]
)
async def unload_model(
    request: Request,
    unload_request: ModelUnloadRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Unload a model from memory.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "unload_model_requested",
        request_id=request_id,
        model_id=unload_request.model_id,
        force=unload_request.force
    )
    
    try:
        # Check if model exists
        model = await model_registry.get_model(unload_request.model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ModelNotFound",
                    "message": f"Model '{unload_request.model_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Check if model is loaded
        if not await model_manager.is_loaded(unload_request.model_id):
            return {
                "status": "not_loaded",
                "model_id": unload_request.model_id,
                "message": f"Model '{unload_request.model_id}' is not loaded",
                "request_id": request_id
            }
        
        # Unload model
        task_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            _unload_model_task,
            task_id=task_id,
            request_id=request_id,
            model_id=unload_request.model_id,
            force=unload_request.force
        )
        
        response = {
            "status": "unloading",
            "task_id": task_id,
            "model_id": unload_request.model_id,
            "message": f"Unloading model '{unload_request.model_id}' in background",
            "request_id": request_id,
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        logger.info(
            "unload_model_accepted",
            request_id=request_id,
            model_id=unload_request.model_id,
            task_id=task_id
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "unload_model_failed",
            request_id=request_id,
            model_id=unload_request.model_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to unload model '{unload_request.model_id}'",
                "request_id": request_id
            }
        )


async def _unload_model_task(
    task_id: str,
    request_id: str,
    model_id: str,
    force: bool
):
    """
    Background task for unloading models.
    """
    logger.info(
        "unload_model_task_started",
        task_id=task_id,
        request_id=request_id,
        model_id=model_id
    )
    
    try:
        # Unload the model
        success = await model_loader.unload_model(
            model_id=model_id,
            force=force
        )
        
        if success:
            # Update model status
            await model_registry.update_model_status(
                model_id,
                "unloaded",
                is_loaded=False,
                loaded_at=None,
                memory_usage_mb=0
            )
            
            # Free memory
            from models.memory_manager import MemoryPool
            memory_pool = MemoryPool()
            memory_pool.deallocate(model_id)
            
            logger.info(
                "unload_model_task_completed",
                task_id=task_id,
                request_id=request_id,
                model_id=model_id
            )
        else:
            logger.error(
                "unload_model_task_failed",
                task_id=task_id,
                request_id=request_id,
                model_id=model_id,
                error="Failed to unload model"
            )
            
    except Exception as e:
        logger.error(
            "unload_model_task_exception",
            task_id=task_id,
            request_id=request_id,
            model_id=model_id,
            error=str(e),
            exc_info=True
        )


@router.get(
    "/task/{task_id}",
    summary="Get Task Status",
    description="""
    Get the status of a background model loading/unloading task.
    
    Returns:
    - Task status (pending, running, completed, failed)
    - Progress percentage
    - Estimated time remaining
    - Error details if failed
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def get_task_status(
    request: Request,
    task_id: str,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Get the status of a background task.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # TODO: Implement task status tracking
    # This would query a task queue/database for status
    
    return {
        "task_id": task_id,
        "status": "completed",  # Placeholder
        "progress": 100,
        "request_id": request_id
    }


# ============================================================================
# MODEL TESTING AND COMPARISON
# ============================================================================

@router.post(
    "/test",
    summary="Test Model",
    description="""
    Test a model with a sample prompt.
    
    This endpoint:
    1. Loads model if not already loaded
    2. Runs inference with test prompt
    3. Returns response with performance metrics
    4. Does NOT cache responses
    5. Useful for benchmarking and debugging
    
    Rate limited: 10 requests per minute
    """,
    response_model=ModelTestResponse,
    dependencies=[Depends(rate_limit(limit=10, period=60))]
)
async def test_model(
    request: Request,
    test_request: ModelTestRequest,
    api_key: str = Depends(verify_api_key)
) -> ModelTestResponse:
    """
    Test a model with a sample prompt.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "test_model_requested",
        request_id=request_id,
        model_id=test_request.model_id,
        prompt_length=len(test_request.prompt)
    )
    
    try:
        # Check if model exists
        model = await model_registry.get_model(test_request.model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ModelNotFound",
                    "message": f"Model '{test_request.model_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Ensure model is loaded
        if not await model_manager.is_loaded(test_request.model_id):
            # Try to load model
            success = await model_loader.load_model(
                model_id=test_request.model_id,
                device="auto"
            )
            
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "ModelNotLoaded",
                        "message": f"Failed to load model '{test_request.model_id}'",
                        "request_id": request_id
                    }
                )
        
        # Get model backend
        backend = await model_manager.get_backend(test_request.model_id)
        
        # Run inference
        inference_start = time.time()
        response = await backend.generate(
            prompt=test_request.prompt,
            max_tokens=test_request.max_tokens,
            temperature=test_request.temperature
        )
        inference_time = time.time() - inference_start
        
        # Calculate metrics
        tokens_generated = len(response.split())
        tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
        
        result = ModelTestResponse(
            model_id=test_request.model_id,
            prompt=test_request.prompt,
            response=response,
            latency_ms=round(inference_time * 1000, 2),
            tokens_generated=tokens_generated,
            tokens_per_second=round(tokens_per_second, 2),
            success=True
        )
        
        logger.info(
            "test_model_completed",
            request_id=request_id,
            model_id=test_request.model_id,
            latency_ms=result.latency_ms,
            tokens_generated=result.tokens_generated,
            tokens_per_second=result.tokens_per_second
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "test_model_failed",
            request_id=request_id,
            model_id=test_request.model_id,
            error=str(e),
            exc_info=True
        )
        
        return ModelTestResponse(
            model_id=test_request.model_id,
            prompt=test_request.prompt,
            response="",
            latency_ms=round((time.time() - start_time) * 1000, 2),
            tokens_generated=0,
            tokens_per_second=0,
            success=False,
            error=str(e)
        )


@router.post(
    "/compare",
    summary="Compare Models",
    description="""
    Compare multiple models side by side with the same prompt.
    
    This endpoint:
    1. Runs the same prompt through multiple models
    2. Returns responses with performance metrics
    3. Provides recommendations based on quality/speed/cost
    4. Perfect for model selection and A/B testing
    
    Rate limited: 5 requests per minute
    """,
    response_model=ModelComparisonResponse,
    dependencies=[Depends(rate_limit(limit=5, period=60))]
)
async def compare_models(
    request: Request,
    comparison_request: ModelComparisonRequest,
    api_key: str = Depends(verify_api_key)
) -> ModelComparisonResponse:
    """
    Compare multiple models with the same prompt.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "compare_models_requested",
        request_id=request_id,
        model_count=len(comparison_request.model_ids),
        prompt_length=len(comparison_request.prompt)
    )
    
    try:
        # Run all models in parallel
        tasks = []
        for model_id in comparison_request.model_ids:
            tasks.append(
                _test_single_model(
                    model_id=model_id,
                    prompt=comparison_request.prompt,
                    max_tokens=comparison_request.max_tokens
                )
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        model_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                model_results.append({
                    "model_id": comparison_request.model_ids[i],
                    "success": False,
                    "error": str(result)
                })
            else:
                model_results.append(result)
        
        # Generate recommendations
        recommendations = []
        
        # Find fastest model
        successful_results = [r for r in model_results if r.get("success")]
        if successful_results:
            fastest = min(successful_results, key=lambda x: x["latency_ms"])
            recommendations.append(f"Fastest: {fastest['model_id']} ({fastest['latency_ms']}ms)")
            
            # Find most tokens per second
            highest_tps = max(successful_results, key=lambda x: x["tokens_per_second"])
            recommendations.append(f"Highest throughput: {highest_tps['model_id']} ({highest_tps['tokens_per_second']} t/s)")
        
        response = ModelComparisonResponse(
            prompt=comparison_request.prompt,
            timestamp=datetime.now(),
            results=model_results,
            recommendations=recommendations
        )
        
        logger.info(
            "compare_models_completed",
            request_id=request_id,
            successful=len(successful_results) if successful_results else 0,
            total=len(comparison_request.model_ids)
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "compare_models_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to compare models",
                "request_id": request_id
            }
        )


async def _test_single_model(
    model_id: str,
    prompt: str,
    max_tokens: int
) -> Dict[str, Any]:
    """
    Test a single model for comparison.
    """
    try:
        start_time = time.time()
        
        # Ensure model is loaded
        if not await model_manager.is_loaded(model_id):
            await model_loader.load_model(model_id, device="auto")
        
        # Get backend
        backend = await model_manager.get_backend(model_id)
        
        # Run inference
        response = await backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        inference_time = time.time() - start_time
        
        tokens_generated = len(response.split())
        tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
        
        return {
            "model_id": model_id,
            "success": True,
            "response": response[:500] + "..." if len(response) > 500 else response,
            "latency_ms": round(inference_time * 1000, 2),
            "tokens_generated": tokens_generated,
            "tokens_per_second": round(tokens_per_second, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "model_id": model_id,
            "success": False,
            "error": str(e)
        }


# ============================================================================
# MODEL METRICS AND MONITORING
# ============================================================================

@router.get(
    "/{model_id}/metrics",
    summary="Get Model Metrics",
    description="""
    Get performance metrics for a specific model.
    
    Returns:
    - Request volume over time
    - Latency percentiles (p50, p95, p99)
    - Error rates
    - Token usage
    - Memory consumption
    - Cost analysis
    
    Supports time range filtering.
    """,
    response_model=ModelMetricsResponse,
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def get_model_metrics(
    request: Request,
    model_id: str,
    hours: int = Query(24, description="Hours of history to include", ge=1, le=168),
    api_key: str = Depends(verify_api_key)
) -> ModelMetricsResponse:
    """
    Get performance metrics for a model.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "get_model_metrics_requested",
        request_id=request_id,
        model_id=model_id,
        hours=hours
    )
    
    try:
        # Check if model exists
        model = await model_registry.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ModelNotFound",
                    "message": f"Model '{model_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Get metrics from collector
        metrics = await metrics_collector.get_model_metrics(model_id, hours=hours)
        
        # Get current memory usage
        memory_usage = await model_manager.get_model_memory_usage(model_id)
        
        # Get GPU usage if available
        gpu_usage = None
        if await model_manager.is_loaded(model_id):
            backend = await model_manager.get_backend(model_id)
            gpu_usage = await backend.get_gpu_usage()
        
        response = ModelMetricsResponse(
            model_id=model_id,
            timestamp=datetime.now(),
            request_count=metrics.get("request_count", 0),
            success_count=metrics.get("success_count", 0),
            error_count=metrics.get("error_count", 0),
            avg_latency_ms=metrics.get("avg_latency_ms", 0),
            p95_latency_ms=metrics.get("p95_latency_ms", 0),
            p99_latency_ms=metrics.get("p99_latency_ms", 0),
            total_tokens=metrics.get("total_tokens", 0),
            avg_tokens_per_request=metrics.get("avg_tokens_per_request", 0),
            tokens_per_second=metrics.get("tokens_per_second", 0),
            memory_usage_mb=memory_usage,
            gpu_usage_percent=gpu_usage,
            cost_total=metrics.get("cost_total", 0),
            cost_per_request=metrics.get("cost_per_request", 0)
        )
        
        logger.info(
            "get_model_metrics_completed",
            request_id=request_id,
            model_id=model_id,
            request_count=response.request_count
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "get_model_metrics_failed",
            request_id=request_id,
            model_id=model_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to get metrics for model '{model_id}'",
                "request_id": request_id
            }
        )


# ============================================================================
# MODEL CONFIGURATION AND MANAGEMENT (ADMIN ONLY)
# ============================================================================

@router.put(
    "/{model_id}",
    summary="Update Model",
    description="""
    Update model configuration.
    
    Admin only endpoint for:
    - Updating model metadata
    - Modifying capabilities
    - Setting fallback models
    - Enabling/disabling models
    
    Rate limited: 5 requests per minute
    """,
    response_model=ModelDetailInfo,
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=5, period=60))]
)
async def update_model(
    request: Request,
    model_id: str,
    update_request: ModelUpdateRequest,
    api_key: str = Depends(verify_api_key)
) -> ModelDetailInfo:
    """
    Update model configuration (admin only).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "update_model_requested",
        request_id=request_id,
        model_id=model_id,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        # Check if model exists
        model = await model_registry.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ModelNotFound",
                    "message": f"Model '{model_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Update model in registry
        updated_model = await model_registry.update_model(
            model_id=model_id,
            name=update_request.name,
            capabilities=update_request.capabilities,
            fallback_models=update_request.fallback_models,
            metadata=update_request.metadata,
            is_active=update_request.is_active
        )
        
        # If model is unloaded, update status
        if update_request.is_active is False:
            if await model_manager.is_loaded(model_id):
                await model_loader.unload_model(model_id)
        
        logger.info(
            "update_model_completed",
            request_id=request_id,
            model_id=model_id
        )
        
        # Return updated model
        return await get_model(request, model_id, api_key)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "update_model_failed",
            request_id=request_id,
            model_id=model_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to update model '{model_id}'",
                "request_id": request_id
            }
        )


@router.delete(
    "/{model_id}",
    summary="Delete Model",
    description="""
    Delete a model from the registry.
    
    Admin only endpoint:
    - Unloads model if loaded
    - Removes from registry
    - Does NOT delete model files
    
    Rate limited: 5 requests per minute
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=5, period=60))]
)
async def delete_model(
    request: Request,
    model_id: str,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Delete a model from registry (admin only).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "delete_model_requested",
        request_id=request_id,
        model_id=model_id,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        # Check if model exists
        model = await model_registry.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ModelNotFound",
                    "message": f"Model '{model_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Unload if loaded
        if await model_manager.is_loaded(model_id):
            await model_loader.unload_model(model_id)
        
        # Delete from registry
        await model_registry.delete_model(model_id)
        
        logger.info(
            "delete_model_completed",
            request_id=request_id,
            model_id=model_id
        )
        
        return {
            "status": "deleted",
            "model_id": model_id,
            "message": f"Model '{model_id}' deleted successfully",
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "delete_model_failed",
            request_id=request_id,
            model_id=model_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to delete model '{model_id}'",
                "request_id": request_id
            }
        )


@router.post(
    "/reload-config",
    summary="Reload Model Config",
    description="""
    Reload model configuration from YAML file.
    
    Admin only endpoint:
    - Reads models.yaml configuration
    - Updates registry with new models
    - Preserves existing loaded models
    
    Rate limited: 2 requests per minute
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=2, period=60))]
)
async def reload_model_config(
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Reload model configuration from YAML file (admin only).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "reload_model_config_requested",
        request_id=request_id,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        # Reload registry from config
        await model_registry.load_from_config(settings.model.config_path)
        
        # Get updated stats
        stats = model_registry.get_stats()
        
        logger.info(
            "reload_model_config_completed",
            request_id=request_id,
            total_models=stats["total"]
        )
        
        return {
            "status": "reloaded",
            "total_models": stats["total"],
            "message": "Model configuration reloaded successfully",
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(
            "reload_model_config_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to reload model configuration",
                "request_id": request_id
            }
        )


# ============================================================================
# MODEL DOWNLOAD MANAGEMENT
# ============================================================================

@router.post(
    "/{model_id}/download",
    summary="Download Model",
    description="""
    Download model files from HuggingFace or other sources.
    
    Admin only endpoint:
    - Downloads model files to local storage
    - Verifies checksums
    - Prepares model for loading
    - Updates download status
    
    Rate limited: 2 requests per minute
    """,
    response_model=Dict[str, Any],
    status_code=202,
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=2, period=60))]
)
async def download_model(
    request: Request,
    model_id: str,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Download model files (admin only).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "download_model_requested",
        request_id=request_id,
        model_id=model_id,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        # Check if model exists
        model = await model_registry.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ModelNotFound",
                    "message": f"Model '{model_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Check if model requires download
        if model.get("type") == "external":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ExternalModel",
                    "message": "External API models do not require download",
                    "request_id": request_id
                }
            )
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Add download task
        background_tasks.add_task(
            _download_model_task,
            task_id=task_id,
            request_id=request_id,
            model_id=model_id,
            model_config=model
        )
        
        return {
            "status": "downloading",
            "task_id": task_id,
            "model_id": model_id,
            "message": f"Downloading model '{model_id}' in background",
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "download_model_failed",
            request_id=request_id,
            model_id=model_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to download model '{model_id}'",
                "request_id": request_id
            }
        )


async def _download_model_task(
    task_id: str,
    request_id: str,
    model_id: str,
    model_config: Dict[str, Any]
):
    """
    Background task for downloading models.
    """
    logger.info(
        "download_model_task_started",
        task_id=task_id,
        request_id=request_id,
        model_id=model_id
    )
    
    try:
        from scripts.download_models import ModelDownloader
        
        downloader = ModelDownloader()
        success = await downloader.download_model(model_id, model_config)
        
        if success:
            await model_registry.update_model_status(
                model_id,
                "downloaded",
                metadata={"downloaded_at": datetime.now().isoformat()}
            )
            
            logger.info(
                "download_model_task_completed",
                task_id=task_id,
                request_id=request_id,
                model_id=model_id
            )
        else:
            logger.error(
                "download_model_task_failed",
                task_id=task_id,
                request_id=request_id,
                model_id=model_id,
                error="Download failed"
            )
            
    except Exception as e:
        logger.error(
            "download_model_task_exception",
            task_id=task_id,
            request_id=request_id,
            model_id=model_id,
            error=str(e),
            exc_info=True
        )


# ============================================================================
# MODEL QUANTIZATION
# ============================================================================

@router.post(
    "/{model_id}/quantize",
    summary="Quantize Model",
    description="""
    Quantize a model to reduce memory usage.
    
    Admin only endpoint:
    - Converts model to 4-bit or 8-bit precision
    - Reduces memory footprint by 4-8x
    - Slightly reduces quality
    - Creates new quantized version
    
    Rate limited: 2 requests per minute
    """,
    response_model=Dict[str, Any],
    status_code=202,
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=2, period=60))]
)
async def quantize_model(
    request: Request,
    model_id: str,
    quantization: str = Query("4bit", description="Quantization level (4bit, 8bit)"),
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Quantize a model (admin only).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "quantize_model_requested",
        request_id=request_id,
        model_id=model_id,
        quantization=quantization,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        # Check if model exists
        model = await model_registry.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ModelNotFound",
                    "message": f"Model '{model_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Check if model can be quantized
        if model.get("type") == "external":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ExternalModel",
                    "message": "External API models cannot be quantized",
                    "request_id": request_id
                }
            )
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Add quantization task
        background_tasks.add_task(
            _quantize_model_task,
            task_id=task_id,
            request_id=request_id,
            model_id=model_id,
            quantization=quantization
        )
        
        return {
            "status": "quantizing",
            "task_id": task_id,
            "model_id": model_id,
            "quantization": quantization,
            "message": f"Quantizing model '{model_id}' to {quantization} in background",
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "quantize_model_failed",
            request_id=request_id,
            model_id=model_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to quantize model '{model_id}'",
                "request_id": request_id
            }
        )


async def _quantize_model_task(
    task_id: str,
    request_id: str,
    model_id: str,
    quantization: str
):
    """
    Background task for quantizing models.
    """
    logger.info(
        "quantize_model_task_started",
        task_id=task_id,
        request_id=request_id,
        model_id=model_id,
        quantization=quantization
    )
    
    try:
        success = await quantization_manager.quantize_model(
            model_id=model_id,
            quantization=quantization
        )
        
        if success:
            logger.info(
                "quantize_model_task_completed",
                task_id=task_id,
                request_id=request_id,
                model_id=model_id,
                quantization=quantization
            )
        else:
            logger.error(
                "quantize_model_task_failed",
                task_id=task_id,
                request_id=request_id,
                model_id=model_id,
                error="Quantization failed"
            )
            
    except Exception as e:
        logger.error(
            "quantize_model_task_exception",
            task_id=task_id,
            request_id=request_id,
            model_id=model_id,
            error=str(e),
            exc_info=True
        )


# ============================================================================
# BULK MODEL OPERATIONS
# ============================================================================

@router.post(
    "/load-all",
    summary="Load All Models",
    description="""
    Load all available models into memory.
    
    Admin only endpoint:
    - Loads all models sequentially
    - Monitors memory usage
    - Useful for pre-warming
    
    Rate limited: 1 request per minute
    """,
    response_model=Dict[str, Any],
    status_code=202,
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=1, period=60))]
)
async def load_all_models(
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Load all available models (admin only).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "load_all_models_requested",
        request_id=request_id,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        task_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            _load_all_models_task,
            task_id=task_id,
            request_id=request_id
        )
        
        return {
            "status": "loading",
            "task_id": task_id,
            "message": "Loading all models in background",
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(
            "load_all_models_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to load all models",
                "request_id": request_id
            }
        )


async def _load_all_models_task(
    task_id: str,
    request_id: str
):
    """
    Background task for loading all models.
    """
    logger.info(
        "load_all_models_task_started",
        task_id=task_id,
        request_id=request_id
    )
    
    try:
        all_models = await model_registry.get_all_models()
        loaded = 0
        failed = 0
        
        for model in all_models:
            if model.get("type") != "external" and model.get("status") == "available":
                try:
                    success = await model_loader.load_model(model["id"])
                    if success:
                        loaded += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                
                # Small delay to prevent overwhelming system
                await asyncio.sleep(0.5)
        
        logger.info(
            "load_all_models_task_completed",
            task_id=task_id,
            request_id=request_id,
            loaded=loaded,
            failed=failed
        )
        
    except Exception as e:
        logger.error(
            "load_all_models_task_failed",
            task_id=task_id,
            request_id=request_id,
            error=str(e),
            exc_info=True
        )


@router.post(
    "/unload-all",
    summary="Unload All Models",
    description="""
    Unload all models from memory.
    
    Admin only endpoint:
    - Unloads all loaded models
    - Frees GPU/CPU memory
    - Emergency resource cleanup
    
    Rate limited: 1 request per minute
    """,
    response_model=Dict[str, Any],
    status_code=202,
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=1, period=60))]
)
async def unload_all_models(
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Unload all models from memory (admin only).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "unload_all_models_requested",
        request_id=request_id,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        task_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            _unload_all_models_task,
            task_id=task_id,
            request_id=request_id
        )
        
        return {
            "status": "unloading",
            "task_id": task_id,
            "message": "Unloading all models in background",
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(
            "unload_all_models_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to unload all models",
                "request_id": request_id
            }
        )


async def _unload_all_models_task(
    task_id: str,
    request_id: str
):
    """
    Background task for unloading all models.
    """
    logger.info(
        "unload_all_models_task_started",
        task_id=task_id,
        request_id=request_id
    )
    
    try:
        loaded_models = await model_manager.get_loaded_models()
        unloaded = 0
        failed = 0
        
        for model in loaded_models:
            try:
                success = await model_loader.unload_model(model["id"])
                if success:
                    unloaded += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
        
        logger.info(
            "unload_all_models_task_completed",
            task_id=task_id,
            request_id=request_id,
            unloaded=unloaded,
            failed=failed
        )
        
    except Exception as e:
        logger.error(
            "unload_all_models_task_failed",
            task_id=task_id,
            request_id=request_id,
            error=str(e),
            exc_info=True
        )


# ============================================================================
# MODEL CAPABILITIES AND DISCOVERY
# ============================================================================

@router.get(
    "/capabilities/list",
    summary="List Capabilities",
    description="""
    List all available model capabilities.
    
    Returns all capability types that can be used for filtering:
    - chat, instruction, reasoning, creative, code, math, vision, etc.
    
    Useful for frontend filter generation.
    """,
    response_model=Dict[str, List[str]],
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def list_capabilities(
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, List[str]]:
    """
    List all available model capabilities.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    all_capabilities = set()
    models = await model_registry.get_all_models()
    
    for model in models:
        capabilities = model.get("capabilities", [])
        all_capabilities.update(capabilities)
    
    return {
        "capabilities": sorted(list(all_capabilities)),
        "count": len(all_capabilities),
        "request_id": request_id
    }


@router.get(
    "/providers/list",
    summary="List Providers",
    description="""
    List all model providers.
    
    Returns all providers with models in the registry:
    - Meta, Mistral, OpenAI, Anthropic, Cohere, xAI, etc.
    
    Useful for frontend filter generation.
    """,
    response_model=Dict[str, List[str]],
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def list_providers(
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, List[str]]:
    """
    List all model providers.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    all_providers = set()
    models = await model_registry.get_all_models()
    
    for model in models:
        provider = model.get("provider", "Unknown")
        all_providers.add(provider)
    
    return {
        "providers": sorted(list(all_providers)),
        "count": len(all_providers),
        "request_id": request_id
    }


@router.get(
    "/stats",
    summary="Model Statistics",
    description="""
    Get aggregate statistics about all models.
    
    Returns:
    - Total models count
    - Models by type
    - Models by status
    - Models by provider
    - Loaded models count
    - Total memory usage
    
    Rate limited: 30 requests per minute
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def get_model_stats(
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Get aggregate model statistics.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        stats = model_registry.get_stats()
        
        # Add real-time stats
        loaded_models = await model_manager.get_loaded_models()
        stats["loaded_count"] = len(loaded_models)
        stats["loaded_models"] = [m["id"] for m in loaded_models]
        
        # Calculate total memory usage
        total_memory_mb = 0
        for model in loaded_models:
            memory = await model_manager.get_model_memory_usage(model["id"])
            if memory:
                total_memory_mb += memory
        
        stats["total_memory_usage_mb"] = round(total_memory_mb, 2)
        
        # Get error rates
        error_models = await model_registry.get_models_by_status("error")
        stats["error_count"] = len(error_models)
        
        return {
            **stats,
            "request_id": request_id,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(
            "get_model_stats_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to get model statistics",
                "request_id": request_id
            }
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "router",
    "ModelInfo",
    "ModelDetailInfo",
    "ModelLoadRequest",
    "ModelUnloadRequest",
    "ModelUpdateRequest",
    "ModelTestRequest",
    "ModelTestResponse",
    "ModelMetricsResponse",
    "ModelComparisonRequest",
    "ModelComparisonResponse"
]