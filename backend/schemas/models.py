"""
Model Schemas - Production Ready
Pydantic models for model management, registration, and monitoring with comprehensive
validation, type safety, and API consistency.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import re

from core.exceptions import ValidationError

# ============================================================================
# ENUMS
# ============================================================================

class ModelStatus(str, Enum):
    """Model status enumeration."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"
    DISABLED = "disabled"


class ModelType(str, Enum):
    """Model type enumeration."""
    LLAMA = "llama"
    MISTRAL = "mistral"
    FALCON = "falcon"
    GPT = "gpt"
    CLAUDE = "claude"
    COHERE = "cohere"
    GROK = "grok"
    QWEN = "qwen"
    PHI = "phi"
    DEEPSEEK = "deepseek"
    STARCODER = "starcoder"
    EXTERNAL = "external"
    MOCK = "mock"


class ModelLibrary(str, Enum):
    """Model library enumeration."""
    TRANSFORMERS = "transformers"
    LLAMA_CPP = "llama-cpp"
    VLLM = "vllm"
    CTRANFORMERS = "ctransformers"
    EXLLAMA = "exllama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GROK = "grok"
    MOCK = "mock"


class QuantizationMethod(str, Enum):
    """Quantization method enumeration."""
    BITS_4 = "4bit"
    BITS_8 = "8bit"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    Q5_0 = "q5_0"
    Q5_1 = "q5_1"
    Q8_0 = "q8_0"
    Q2_K = "q2_k"
    Q3_K = "q3_k"
    Q4_K = "q4_k"
    Q5_K = "q5_k"
    Q6_K = "q6_k"
    Q8_K = "q8_k"


class ModelFormat(str, Enum):
    """Model file format enumeration."""
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    GGUF = "gguf"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    API = "api"  # External API model


# ============================================================================
# MODEL REGISTRATION SCHEMAS
# ============================================================================

class ModelCapability(BaseModel):
    """Model capability with confidence score."""
    
    name: str = Field(
        ...,
        description="Capability name",
        example="code"
    )
    confidence: float = Field(
        1.0,
        description="Confidence score for this capability",
        ge=0.0,
        le=1.0,
        example=0.95
    )
    
    @validator('name')
    def validate_name(cls, v):
        """Validate capability name."""
        allowed = [
            "chat", "code", "reasoning", "creative", "instruction",
            "qa", "summarization", "translation", "analysis",
            "multilingual", "vision", "audio", "embedding",
            "general", "fast", "lightweight", "technical"
        ]
        if v not in allowed:
            raise ValueError(f"Capability must be one of {allowed}")
        return v


class ModelDownloadConfig(BaseModel):
    """Model download configuration."""
    
    source: str = Field(
        ...,
        description="Download source (huggingface, direct, s3, gcs)",
        example="huggingface"
    )
    repo_id: Optional[str] = Field(
        None,
        description="HuggingFace repository ID",
        example="meta-llama/Llama-2-7b-chat-hf"
    )
    filename: Optional[str] = Field(
        None,
        description="Specific filename to download",
        example="llama-2-7b-chat.Q4_K_M.gguf"
    )
    url: Optional[str] = Field(
        None,
        description="Direct download URL",
        example="https://example.com/model.gguf"
    )
    file_size_mb: Optional[float] = Field(
        None,
        description="Expected file size in MB",
        gt=0,
        example=4760.5
    )
    checksum: Optional[str] = Field(
        None,
        description="File checksum (format: algorithm:hash)",
        example="sha256:abc123..."
    )
    
    @validator('source')
    def validate_source(cls, v):
        """Validate download source."""
        allowed = ["huggingface", "direct", "s3", "gcs", "azure"]
        if v not in allowed:
            raise ValueError(f"Source must be one of {allowed}")
        return v
    
    @validator('repo_id')
    def validate_repo_id(cls, v, values):
        """Validate repo_id based on source."""
        if values.get('source') == 'huggingface' and not v:
            raise ValueError("repo_id required for HuggingFace source")
        return v
    
    @validator('url')
    def validate_url(cls, v, values):
        """Validate URL based on source."""
        if values.get('source') == 'direct' and not v:
            raise ValueError("URL required for direct download")
        if v and not re.match(r'^https?://', v):
            raise ValueError("URL must start with http:// or https://")
        return v
    
    @validator('checksum')
    def validate_checksum(cls, v):
        """Validate checksum format."""
        if v:
            if ':' not in v:
                raise ValueError("Checksum must be in format algorithm:hash")
            algorithm, hash_value = v.split(':', 1)
            if algorithm not in ['md5', 'sha1', 'sha256', 'sha512']:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            if not re.match(r'^[a-fA-F0-9]+$', hash_value):
                raise ValueError("Hash must be hexadecimal")
        return v


class ModelResourceRequirements(BaseModel):
    """Model resource requirements."""
    
    memory_gb: float = Field(
        ...,
        description="Minimum RAM required in GB",
        gt=0,
        example=8.0
    )
    gpu_memory_gb: Optional[float] = Field(
        None,
        description="Minimum GPU memory required in GB",
        gt=0,
        example=6.0
    )
    min_cpu_cores: int = Field(
        2,
        description="Minimum CPU cores required",
        ge=1,
        le=64,
        example=4
    )
    recommended_batch_size: int = Field(
        1,
        description="Recommended batch size for optimal performance",
        ge=1,
        le=128,
        example=4
    )
    
    @root_validator
    def validate_memory_requirements(cls, values):
        """Validate memory requirements consistency."""
        gpu_memory = values.get('gpu_memory_gb')
        memory = values.get('memory_gb', 0)
        
        if gpu_memory and gpu_memory > memory:
            raise ValueError("GPU memory cannot exceed system memory")
        
        return values


class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics for comparison."""
    
    expected_latency_ms: int = Field(
        ...,
        description="Expected latency in milliseconds",
        gt=0,
        le=10000,
        example=120
    )
    tokens_per_second: float = Field(
        ...,
        description="Expected tokens per second",
        gt=0,
        le=1000,
        example=45.5
    )
    quality_score: float = Field(
        0.7,
        description="Quality score (0-1)",
        ge=0.0,
        le=1.0,
        example=0.85
    )
    reliability_score: float = Field(
        0.95,
        description="Reliability score (0-1)",
        ge=0.0,
        le=1.0,
        example=0.98
    )
    
    @validator('quality_score')
    def validate_quality_score(cls, v):
        """Validate quality score."""
        if v < 0 or v > 1:
            raise ValueError("Quality score must be between 0 and 1")
        return v


class ModelQuantizationOption(BaseModel):
    """Available quantization option for a model."""
    
    method: QuantizationMethod = Field(
        ...,
        description="Quantization method"
    )
    memory_gb: float = Field(
        ...,
        description="Memory required with this quantization",
        gt=0,
        example=4.0
    )
    quality_impact: float = Field(
        0.0,
        description="Quality impact relative to full precision (0-1)",
        ge=0.0,
        le=1.0,
        example=0.1
    )
    recommended: bool = Field(
        False,
        description="Whether this quantization is recommended"
    )


class ModelRegistrationRequest(BaseModel):
    """
    Model registration request schema.
    
    Used for registering new models in the system.
    """
    
    model_id: str = Field(
        ...,
        description="Unique model identifier",
        min_length=3,
        max_length=100,
        example="llama-2-7b-chat"
    )
    name: str = Field(
        ...,
        description="Human-readable model name",
        min_length=3,
        max_length=200,
        example="Llama 2 7B Chat"
    )
    provider: str = Field(
        ...,
        description="Model provider",
        min_length=2,
        max_length=100,
        example="Meta"
    )
    type: ModelType = Field(
        ...,
        description="Model type"
    )
    library: ModelLibrary = Field(
        ...,
        description="Inference library"
    )
    format: ModelFormat = Field(
        ...,
        description="Model file format"
    )
    quantization: Optional[QuantizationMethod] = Field(
        None,
        description="Quantization method"
    )
    context_size: int = Field(
        2048,
        description="Maximum context window size",
        ge=512,
        le=131072,
        example=4096
    )
    capabilities: List[Union[str, ModelCapability]] = Field(
        ...,
        description="Model capabilities",
        min_items=1
    )
    
    # Download configuration
    download: ModelDownloadConfig = Field(
        ...,
        description="Model download configuration"
    )
    
    # Resource requirements
    resources: ModelResourceRequirements = Field(
        ...,
        description="Resource requirements"
    )
    
    # Performance metrics
    performance: ModelPerformanceMetrics = Field(
        ...,
        description="Expected performance metrics"
    )
    
    # Quantization options
    quantization_options: List[ModelQuantizationOption] = Field(
        default_factory=list,
        description="Available quantization options"
    )
    
    # Fallback models
    fallback_models: List[str] = Field(
        default_factory=list,
        description="Fallback model IDs"
    )
    
    # Access control
    access_tier: str = Field(
        "free",
        description="Access tier (free, pro, enterprise)",
        example="pro"
    )
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata"
    )
    
    @validator('model_id')
    def validate_model_id(cls, v):
        """Validate model ID format."""
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$', v):
            raise ValueError(
                "Model ID must start with alphanumeric and contain only "
                "alphanumeric, underscore, hyphen, and dot"
            )
        return v.lower()
    
    @validator('fallback_models', each_item=True)
    def validate_fallback_model(cls, v):
        """Validate fallback model ID format."""
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$', v):
            raise ValueError(f"Invalid fallback model ID: {v}")
        return v.lower()
    
    @validator('access_tier')
    def validate_access_tier(cls, v):
        """Validate access tier."""
        allowed = ["free", "pro", "enterprise", "internal"]
        if v not in allowed:
            raise ValueError(f"Access tier must be one of {allowed}")
        return v
    
    @root_validator
    def validate_quantization(cls, values):
        """Validate quantization consistency."""
        quantization = values.get('quantization')
        format = values.get('format')
        
        if quantization and format == ModelFormat.API:
            raise ValueError("API models cannot be quantized")
        
        return values
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "llama-2-7b-chat",
                "name": "Llama 2 7B Chat",
                "provider": "Meta",
                "type": "llama",
                "library": "transformers",
                "format": "safetensors",
                "quantization": "fp16",
                "context_size": 4096,
                "capabilities": ["chat", "instruction", "reasoning"],
                "download": {
                    "source": "huggingface",
                    "repo_id": "meta-llama/Llama-2-7b-chat-hf",
                    "file_size_mb": 13400
                },
                "resources": {
                    "memory_gb": 14.0,
                    "gpu_memory_gb": 7.0,
                    "min_cpu_cores": 4,
                    "recommended_batch_size": 2
                },
                "performance": {
                    "expected_latency_ms": 250,
                    "tokens_per_second": 35.5,
                    "quality_score": 0.85
                },
                "quantization_options": [
                    {
                        "method": "4bit",
                        "memory_gb": 4.0,
                        "quality_impact": 0.15,
                        "recommended": True
                    }
                ],
                "fallback_models": ["mistral-7b-instruct"],
                "access_tier": "free",
                "metadata": {
                    "hf_id": "meta-llama/Llama-2-7b-chat-hf",
                    "release_date": "2023-07-18",
                    "parameters": "7B",
                    "architecture": "transformer"
                }
            }
        }


class ModelRegistrationResponse(BaseModel):
    """Model registration response schema."""
    
    id: str = Field(
        ...,
        description="Internal database ID"
    )
    model_id: str = Field(
        ...,
        description="Model identifier"
    )
    name: str = Field(
        ...,
        description="Model name"
    )
    provider: str = Field(
        ...,
        description="Model provider"
    )
    type: ModelType = Field(
        ...,
        description="Model type"
    )
    status: ModelStatus = Field(
        ...,
        description="Current status"
    )
    created_at: datetime = Field(
        ...,
        description="Creation timestamp"
    )
    message: str = Field(
        ...,
        description="Registration message"
    )


# ============================================================================
# MODEL LOADING SCHEMAS
# ============================================================================

class ModelLoadRequest(BaseModel):
    """
    Model load request schema.
    
    Used for loading models into memory.
    """
    
    model_id: str = Field(
        ...,
        description="Model ID to load",
        example="llama-2-7b-chat"
    )
    quantization: Optional[QuantizationMethod] = Field(
        None,
        description="Quantization method (overrides default)"
    )
    device: str = Field(
        "auto",
        description="Device to load on (cpu, cuda, mps, auto)",
        example="cuda"
    )
    gpu_layers: Optional[int] = Field(
        None,
        description="Number of layers to offload to GPU (for llama.cpp)",
        ge=0,
        le=128,
        example=32
    )
    batch_size: Optional[int] = Field(
        None,
        description="Batch size for inference",
        ge=1,
        le=32,
        example=4
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum tokens to generate",
        ge=1,
        le=4096,
        example=1000
    )
    temperature: Optional[float] = Field(
        None,
        description="Default temperature",
        ge=0.0,
        le=2.0,
        example=0.7
    )
    
    @validator('device')
    def validate_device(cls, v):
        """Validate device."""
        allowed = ['auto', 'cpu', 'cuda', 'mps']
        if v not in allowed:
            raise ValueError(f"Device must be one of {allowed}")
        return v
    
    @validator('gpu_layers')
    def validate_gpu_layers(cls, v, values):
        """Validate GPU layers based on device."""
        if v and v > 0 and values.get('device') == 'cpu':
            raise ValueError("Cannot use GPU layers with CPU device")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "llama-2-7b-chat",
                "quantization": "4bit",
                "device": "cuda",
                "gpu_layers": 32,
                "batch_size": 4,
                "max_tokens": 2048,
                "temperature": 0.7
            }
        }


class ModelLoadResponse(BaseModel):
    """Model load response schema."""
    
    status: str = Field(
        ...,
        description="Load status",
        example="loading"
    )
    task_id: str = Field(
        ...,
        description="Task ID for tracking",
        example="task-123abc"
    )
    model_id: str = Field(
        ...,
        description="Model ID"
    )
    message: str = Field(
        ...,
        description="Status message"
    )
    estimated_time_seconds: int = Field(
        ...,
        description="Estimated load time in seconds",
        example=30
    )
    request_id: str = Field(
        ...,
        description="Request ID for tracing"
    )


class ModelUnloadRequest(BaseModel):
    """Model unload request schema."""
    
    model_id: str = Field(
        ...,
        description="Model ID to unload"
    )
    force: bool = Field(
        False,
        description="Force unload even if in use"
    )


class ModelUnloadResponse(BaseModel):
    """Model unload response schema."""
    
    status: str = Field(
        ...,
        description="Unload status",
        example="unloaded"
    )
    model_id: str = Field(
        ...,
        description="Model ID"
    )
    message: str = Field(
        ...,
        description="Status message"
    )
    memory_freed_mb: Optional[float] = Field(
        None,
        description="Memory freed in MB",
        example=5120.5
    )


# ============================================================================
# MODEL INFORMATION SCHEMAS
# ============================================================================

class ModelInfo(BaseModel):
    """
    Basic model information schema.
    
    Used for listing models and public endpoints.
    """
    
    id: str = Field(
        ...,
        description="Unique model identifier",
        example="llama-2-7b-chat"
    )
    name: str = Field(
        ...,
        description="Human-readable model name",
        example="Llama 2 7B Chat"
    )
    provider: str = Field(
        ...,
        description="Model provider",
        example="Meta"
    )
    type: ModelType = Field(
        ...,
        description="Model type"
    )
    library: ModelLibrary = Field(
        ...,
        description="Inference library"
    )
    format: Optional[ModelFormat] = Field(
        None,
        description="Model format"
    )
    quantization: Optional[QuantizationMethod] = Field(
        None,
        description="Current quantization"
    )
    context_size: int = Field(
        ...,
        description="Maximum context window size",
        example=4096
    )
    capabilities: List[str] = Field(
        ...,
        description="Model capabilities",
        example=["chat", "code", "reasoning"]
    )
    status: ModelStatus = Field(
        ...,
        description="Current status"
    )
    is_loaded: bool = Field(
        False,
        description="Whether model is currently loaded"
    )
    loaded_at: Optional[datetime] = Field(
        None,
        description="When model was loaded"
    )
    memory_usage_mb: Optional[float] = Field(
        None,
        description="Current memory usage in MB"
    )
    requires_api_key: bool = Field(
        False,
        description="Whether model requires API key"
    )
    api_key_configured: Optional[bool] = Field(
        None,
        description="Whether API key is configured"
    )
    cost_per_token: Optional[float] = Field(
        None,
        description="Cost per token in USD",
        example=0.00001
    )
    latency_p95_ms: Optional[float] = Field(
        None,
        description="95th percentile latency in ms",
        example=120.5
    )
    error_count: int = Field(
        0,
        description="Number of errors encountered"
    )
    total_requests: int = Field(
        0,
        description="Total number of requests"
    )
    created_at: datetime = Field(
        ...,
        description="When model was added to registry"
    )
    updated_at: datetime = Field(
        ...,
        description="When model was last updated"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "llama-2-7b-chat",
                "name": "Llama 2 7B Chat",
                "provider": "Meta",
                "type": "llama",
                "library": "transformers",
                "format": "safetensors",
                "quantization": "fp16",
                "context_size": 4096,
                "capabilities": ["chat", "instruction", "reasoning"],
                "status": "ready",
                "is_loaded": True,
                "loaded_at": "2024-01-15T10:30:00Z",
                "memory_usage_mb": 14336.5,
                "requires_api_key": False,
                "cost_per_token": 0.000002,
                "latency_p95_ms": 245.3,
                "error_count": 3,
                "total_requests": 15234,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }


class ModelDetailInfo(ModelInfo):
    """
    Detailed model information schema.
    
    Used for admin endpoints and detailed views.
    """
    
    config: Dict[str, Any] = Field(
        ...,
        description="Full model configuration"
    )
    download_url: Optional[str] = Field(
        None,
        description="URL to download model"
    )
    file_size_mb: Optional[float] = Field(
        None,
        description="Model file size in MB"
    )
    memory_required_gb: Optional[float] = Field(
        None,
        description="Minimum memory required in GB"
    )
    gpu_memory_required_gb: Optional[float] = Field(
        None,
        description="Minimum GPU memory required in GB"
    )
    recommended_batch_size: Optional[int] = Field(
        None,
        description="Recommended batch size"
    )
    quantization_options: List[ModelQuantizationOption] = Field(
        default_factory=list,
        description="Available quantization options"
    )
    fallback_models: List[str] = Field(
        default_factory=list,
        description="Fallback model IDs"
    )
    access_tier: str = Field(
        "free",
        description="Access tier"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                **ModelInfo.Config.json_schema_extra["example"],
                "config": {
                    "model_id": "llama-2-7b-chat",
                    "hf_id": "meta-llama/Llama-2-7b-chat-hf",
                    "parameters": "7B",
                    "architecture": "transformer"
                },
                "file_size_mb": 13400,
                "memory_required_gb": 14.0,
                "gpu_memory_required_gb": 7.0,
                "recommended_batch_size": 2,
                "quantization_options": [
                    {
                        "method": "4bit",
                        "memory_gb": 4.0,
                        "quality_impact": 0.15,
                        "recommended": True
                    }
                ],
                "fallback_models": ["mistral-7b-instruct"],
                "access_tier": "free",
                "metadata": {
                    "hf_id": "meta-llama/Llama-2-7b-chat-hf",
                    "release_date": "2023-07-18",
                    "parameters": "7B",
                    "architecture": "transformer"
                }
            }
        }


# ============================================================================
# MODEL TESTING SCHEMAS
# ============================================================================

class ModelTestRequest(BaseModel):
    """
    Model test request schema.
    
    Used for testing model performance and quality.
    """
    
    model_id: str = Field(
        ...,
        description="Model ID to test"
    )
    prompt: str = Field(
        ...,
        description="Test prompt",
        min_length=1,
        max_length=1000,
        example="What is the capital of France?"
    )
    max_tokens: int = Field(
        50,
        description="Maximum tokens to generate",
        ge=1,
        le=500,
        example=100
    )
    temperature: float = Field(
        0.7,
        description="Temperature",
        ge=0.0,
        le=2.0,
        example=0.7
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "llama-2-7b-chat",
                "prompt": "Write a Python function to reverse a string",
                "max_tokens": 150,
                "temperature": 0.8
            }
        }


class ModelTestResponse(BaseModel):
    """Model test response schema."""
    
    model_id: str = Field(
        ...,
        description="Model ID tested"
    )
    prompt: str = Field(
        ...,
        description="Test prompt"
    )
    response: str = Field(
        ...,
        description="Model response"
    )
    latency_ms: float = Field(
        ...,
        description="Latency in milliseconds",
        example=450.23
    )
    tokens_generated: int = Field(
        ...,
        description="Number of tokens generated",
        example=120
    )
    tokens_per_second: float = Field(
        ...,
        description="Tokens per second",
        example=26.7
    )
    success: bool = Field(
        ...,
        description="Whether test was successful"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if failed"
    )


class ModelComparisonRequest(BaseModel):
    """
    Model comparison request schema.
    
    Used for comparing multiple models side by side.
    """
    
    model_ids: List[str] = Field(
        ...,
        description="Model IDs to compare",
        min_items=2,
        max_items=5,
        example=["llama-2-7b-chat", "mistral-7b-instruct", "gpt-3.5-turbo"]
    )
    prompt: str = Field(
        ...,
        description="Prompt to test with",
        min_length=1,
        max_length=1000,
        example="Write a haiku about programming"
    )
    max_tokens: int = Field(
        100,
        description="Maximum tokens to generate per model",
        ge=1,
        le=500,
        example=150
    )
    
    @validator('model_ids')
    def validate_model_ids(cls, v):
        """Validate model IDs are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Model IDs must be unique")
        return v


class ModelComparisonResponse(BaseModel):
    """Model comparison response schema."""
    
    prompt: str = Field(
        ...,
        description="Test prompt"
    )
    timestamp: datetime = Field(
        ...,
        description="Comparison timestamp"
    )
    results: List[Dict[str, Any]] = Field(
        ...,
        description="Results for each model"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Model recommendations based on results"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Write a haiku about programming",
                "timestamp": "2024-01-15T10:30:00Z",
                "results": [
                    {
                        "model_id": "llama-2-7b-chat",
                        "success": True,
                        "response": "Silent circuits hum,\nCode flows like water downstream,\nBugs hide in the dark.",
                        "latency_ms": 450,
                        "tokens_per_second": 35.2
                    },
                    {
                        "model_id": "mistral-7b-instruct",
                        "success": True,
                        "response": "Lines of logic dance,\nSyntax errors fade away,\nProgram comes alive.",
                        "latency_ms": 320,
                        "tokens_per_second": 48.7
                    }
                ],
                "recommendations": [
                    "Fastest: mistral-7b-instruct (320ms)",
                    "Highest throughput: mistral-7b-instruct (48.7 t/s)"
                ]
            }
        }


# ============================================================================
# MODEL METRICS SCHEMAS
# ============================================================================

class ModelMetricsResponse(BaseModel):
    """
    Model performance metrics response schema.
    
    Used for monitoring and analytics.
    """
    
    model_id: str = Field(
        ...,
        description="Model ID"
    )
    timestamp: datetime = Field(
        ...,
        description="Metrics timestamp"
    )
    request_count: int = Field(
        ...,
        description="Total requests",
        example=15234
    )
    success_count: int = Field(
        ...,
        description="Successful requests",
        example=15123
    )
    error_count: int = Field(
        ...,
        description="Failed requests",
        example=111
    )
    avg_latency_ms: float = Field(
        ...,
        description="Average latency",
        example=245.3
    )
    p95_latency_ms: float = Field(
        ...,
        description="95th percentile latency",
        example=450.2
    )
    p99_latency_ms: float = Field(
        ...,
        description="99th percentile latency",
        example=890.5
    )
    total_tokens: int = Field(
        ...,
        description="Total tokens generated",
        example=1523400
    )
    avg_tokens_per_request: float = Field(
        ...,
        description="Average tokens per request",
        example=100.2
    )
    tokens_per_second: float = Field(
        ...,
        description="Tokens per second",
        example=35.7
    )
    memory_usage_mb: Optional[float] = Field(
        None,
        description="Current memory usage",
        example=14336.5
    )
    gpu_usage_percent: Optional[float] = Field(
        None,
        description="GPU utilization",
        example=78.5
    )
    cost_total: float = Field(
        0.0,
        description="Total cost in USD",
        example=15.23
    )
    cost_per_request: float = Field(
        0.0,
        description="Average cost per request",
        example=0.001
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "llama-2-7b-chat",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_count": 15234,
                "success_count": 15123,
                "error_count": 111,
                "avg_latency_ms": 245.3,
                "p95_latency_ms": 450.2,
                "p99_latency_ms": 890.5,
                "total_tokens": 1523400,
                "avg_tokens_per_request": 100.2,
                "tokens_per_second": 35.7,
                "memory_usage_mb": 14336.5,
                "gpu_usage_percent": 78.5,
                "cost_total": 15.23,
                "cost_per_request": 0.001
            }
        }


# ============================================================================
# MODEL UPDATE SCHEMAS
# ============================================================================

class ModelUpdateRequest(BaseModel):
    """
    Model update request schema.
    
    Used for updating model configuration.
    """
    
    model_id: str = Field(
        ...,
        description="Model ID to update"
    )
    name: Optional[str] = Field(
        None,
        description="Updated model name",
        min_length=3,
        max_length=200
    )
    capabilities: Optional[List[Union[str, ModelCapability]]] = Field(
        None,
        description="Updated capabilities"
    )
    fallback_models: Optional[List[str]] = Field(
        None,
        description="Updated fallback models"
    )
    access_tier: Optional[str] = Field(
        None,
        description="Updated access tier"
    )
    is_active: Optional[bool] = Field(
        None,
        description="Whether model is active"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated metadata"
    )
    
    @validator('model_id')
    def validate_model_id(cls, v):
        """Validate model ID format."""
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$', v):
            raise ValueError("Invalid model ID format")
        return v.lower()


class ModelUpdateResponse(BaseModel):
    """Model update response schema."""
    
    status: str = Field(
        ...,
        description="Update status",
        example="updated"
    )
    model_id: str = Field(
        ...,
        description="Model ID"
    )
    updated_fields: List[str] = Field(
        ...,
        description="Fields that were updated"
    )
    message: str = Field(
        ...,
        description="Status message"
    )


# ============================================================================
# MODEL STATISTICS SCHEMAS
# ============================================================================

class ModelStatisticsResponse(BaseModel):
    """
    Model statistics response schema.
    
    Aggregated statistics for all models.
    """
    
    total_models: int = Field(
        ...,
        description="Total number of models",
        example=25
    )
    by_status: Dict[str, int] = Field(
        ...,
        description="Models grouped by status",
        example={
            "ready": 12,
            "loading": 2,
            "error": 1,
            "unloaded": 10
        }
    )
    by_type: Dict[str, int] = Field(
        ...,
        description="Models grouped by type",
        example={
            "llama": 8,
            "mistral": 6,
            "external": 11
        }
    )
    by_provider: Dict[str, int] = Field(
        ...,
        description="Models grouped by provider",
        example={
            "Meta": 4,
            "Mistral": 3,
            "OpenAI": 2,
            "Anthropic": 2,
            "Cohere": 1,
            "xAI": 1
        }
    )
    loaded_count: int = Field(
        ...,
        description="Number of loaded models",
        example=5
    )
    error_count: int = Field(
        ...,
        description="Number of models in error state",
        example=1
    )
    total_inferences: int = Field(
        ...,
        description="Total inferences across all models",
        example=1523400
    )
    total_memory_usage_mb: float = Field(
        ...,
        description="Total memory usage in MB",
        example=45678.2
    )
    timestamp: datetime = Field(
        ...,
        description="Statistics timestamp"
    )


# ============================================================================
# MODEL QUANTIZATION SCHEMAS
# ============================================================================

class ModelQuantizeRequest(BaseModel):
    """
    Model quantization request schema.
    
    Used for quantizing models to reduce memory usage.
    """
    
    model_id: str = Field(
        ...,
        description="Model ID to quantize"
    )
    quantization: QuantizationMethod = Field(
        ...,
        description="Quantization method"
    )
    output_path: Optional[str] = Field(
        None,
        description="Output path for quantized model"
    )
    
    @validator('quantization')
    def validate_quantization(cls, v):
        """Validate quantization method is supported."""
        # List of supported quantization methods
        supported = [
            "4bit", "8bit", "fp16", "bf16",
            "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"
        ]
        if v.value not in supported:
            raise ValueError(f"Quantization method {v} not supported")
        return v


class ModelQuantizeResponse(BaseModel):
    """Model quantization response schema."""
    
    status: str = Field(
        ...,
        description="Quantization status",
        example="quantizing"
    )
    task_id: str = Field(
        ...,
        description="Task ID for tracking"
    )
    model_id: str = Field(
        ...,
        description="Original model ID"
    )
    quantization: QuantizationMethod = Field(
        ...,
        description="Quantization method"
    )
    output_model_id: Optional[str] = Field(
        None,
        description="ID of quantized model"
    )
    estimated_time_seconds: int = Field(
        ...,
        description="Estimated quantization time",
        example=300
    )


# ============================================================================
# MODEL DOWNLOAD SCHEMAS
# ============================================================================

class ModelDownloadRequest(BaseModel):
    """
    Model download request schema.
    
    Used for downloading models from HuggingFace or other sources.
    """
    
    model_id: str = Field(
        ...,
        description="Model ID to download"
    )
    force: bool = Field(
        False,
        description="Force re-download even if exists"
    )


class ModelDownloadResponse(BaseModel):
    """Model download response schema."""
    
    status: str = Field(
        ...,
        description="Download status",
        example="downloading"
    )
    task_id: str = Field(
        ...,
        description="Task ID for tracking"
    )
    model_id: str = Field(
        ...,
        description="Model ID"
    )
    progress: Optional[float] = Field(
        None,
        description="Download progress (0-100)",
        example=45.5
    )
    estimated_time_seconds: Optional[int] = Field(
        None,
        description="Estimated remaining time",
        example=120
    )
    file_size_mb: Optional[float] = Field(
        None,
        description="Total file size in MB",
        example=4760.5
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ModelStatus",
    "ModelType",
    "ModelLibrary",
    "QuantizationMethod",
    "ModelFormat",
    
    # Registration
    "ModelCapability",
    "ModelDownloadConfig",
    "ModelResourceRequirements",
    "ModelPerformanceMetrics",
    "ModelQuantizationOption",
    "ModelRegistrationRequest",
    "ModelRegistrationResponse",
    
    # Loading
    "ModelLoadRequest",
    "ModelLoadResponse",
    "ModelUnloadRequest",
    "ModelUnloadResponse",
    
    # Information
    "ModelInfo",
    "ModelDetailInfo",
    
    # Testing
    "ModelTestRequest",
    "ModelTestResponse",
    "ModelComparisonRequest",
    "ModelComparisonResponse",
    
    # Metrics
    "ModelMetricsResponse",
    
    # Update
    "ModelUpdateRequest",
    "ModelUpdateResponse",
    
    # Statistics
    "ModelStatisticsResponse",
    
    # Quantization
    "ModelQuantizeRequest",
    "ModelQuantizeResponse",
    
    # Download
    "ModelDownloadRequest",
    "ModelDownloadResponse"
]