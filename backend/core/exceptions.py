"""
Custom Exceptions - Production Ready
Centralized exception hierarchy for LLM Gateway with detailed error codes,
status mapping, and structured error responses.
"""

from typing import Any, Dict, Optional, List, Union
from fastapi import status
from datetime import datetime
import uuid


class LLMGatewayException(Exception):
    """
    Base exception for all LLM Gateway custom exceptions.
    
    Provides consistent error structure with:
    - HTTP status code
    - Error code
    - Human-readable message
    - Detailed description
    - Request ID for tracing
    - Timestamp
    - Additional context
    
    All other exceptions inherit from this class.
    """
    
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        message: str = "An internal error occurred",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.detail = detail or message
        self.request_id = request_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.context = context or {}
        self.headers = headers or {}
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        error_dict = {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "detail": self.detail,
                "request_id": self.request_id,
                "timestamp": self.timestamp,
                "status_code": self.status_code
            }
        }
        
        if self.context:
            error_dict["error"]["context"] = self.context
        
        return error_dict
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message} (Request: {self.request_id})"


# ============================================================================
# AUTHENTICATION & AUTHORIZATION EXCEPTIONS
# ============================================================================

class AuthenticationError(LLMGatewayException):
    """Raised when authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_FAILED",
            message=message,
            detail=detail or "Invalid or missing authentication credentials",
            request_id=request_id,
            context=context,
            headers={"WWW-Authenticate": "Bearer"}
        )


class InvalidAPIKeyError(AuthenticationError):
    """Raised when API key is invalid."""
    
    def __init__(
        self,
        message: str = "Invalid API key",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            detail=detail or "The provided API key is invalid or expired",
            request_id=request_id,
            context=context
        )
        self.error_code = "INVALID_API_KEY"


class ExpiredAPIKeyError(AuthenticationError):
    """Raised when API key has expired."""
    
    def __init__(
        self,
        message: str = "API key expired",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            detail=detail or "The provided API key has expired",
            request_id=request_id,
            context=context
        )
        self.error_code = "EXPIRED_API_KEY"


class InvalidTokenError(AuthenticationError):
    """Raised when JWT token is invalid."""
    
    def __init__(
        self,
        message: str = "Invalid token",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            detail=detail or "The provided JWT token is invalid or malformed",
            request_id=request_id,
            context=context
        )
        self.error_code = "INVALID_TOKEN"


class ExpiredTokenError(AuthenticationError):
    """Raised when JWT token has expired."""
    
    def __init__(
        self,
        message: str = "Token expired",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            detail=detail or "The provided JWT token has expired",
            request_id=request_id,
            context=context
        )
        self.error_code = "EXPIRED_TOKEN"


class AdminPermissionError(LLMGatewayException):
    """Raised when user lacks admin privileges."""
    
    def __init__(
        self,
        message: str = "Admin access required",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="ADMIN_REQUIRED",
            message=message,
            detail=detail or "This endpoint requires administrator privileges",
            request_id=request_id,
            context=context
        )


class InsufficientPermissionsError(LLMGatewayException):
    """Raised when user lacks specific permissions."""
    
    def __init__(
        self,
        required_permissions: List[str],
        message: str = "Insufficient permissions",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["required_permissions"] = required_permissions
        
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="INSUFFICIENT_PERMISSIONS",
            message=message,
            detail=detail or f"This operation requires: {', '.join(required_permissions)}",
            request_id=request_id,
            context=context
        )


# ============================================================================
# RATE LIMITING EXCEPTIONS
# ============================================================================

class RateLimitExceededError(LLMGatewayException):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        limit: int,
        reset_time: int,
        message: str = "Rate limit exceeded",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["limit"] = limit
        context["reset_time"] = reset_time
        
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(reset_time),
            "Retry-After": str(max(0, reset_time - int(datetime.utcnow().timestamp())))
        }
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            message=message,
            detail=detail or f"Rate limit of {limit} requests exceeded. Try again later.",
            request_id=request_id,
            context=context,
            headers=headers
        )


class ConcurrentRequestLimitError(LLMGatewayException):
    """Raised when concurrent request limit is exceeded."""
    
    def __init__(
        self,
        limit: int,
        current: int,
        message: str = "Concurrent request limit exceeded",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["limit"] = limit
        context["current"] = current
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="CONCURRENT_LIMIT_EXCEEDED",
            message=message,
            detail=detail or f"Maximum concurrent requests ({limit}) exceeded. Currently at {current}.",
            request_id=request_id,
            context=context
        )


# ============================================================================
# MODEL EXCEPTIONS
# ============================================================================

class ModelError(LLMGatewayException):
    """Base class for model-related exceptions."""
    
    def __init__(
        self,
        model_id: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "MODEL_ERROR",
        message: str = "Model error occurred",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["model_id"] = model_id
        
        super().__init__(
            status_code=status_code,
            error_code=error_code,
            message=message,
            detail=detail,
            request_id=request_id,
            context=context
        )


class ModelNotFoundError(ModelError):
    """Raised when a model is not found in registry."""
    
    def __init__(
        self,
        model_id: str,
        message: str = "Model not found",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            model_id=model_id,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="MODEL_NOT_FOUND",
            message=message,
            detail=detail or f"Model '{model_id}' not found in registry",
            request_id=request_id,
            context=context
        )


class ModelNotAvailableError(ModelError):
    """Raised when a model is not available for inference."""
    
    def __init__(
        self,
        model_id: str,
        status: str = "unavailable",
        message: str = "Model not available",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["status"] = status
        
        super().__init__(
            model_id=model_id,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MODEL_NOT_AVAILABLE",
            message=message,
            detail=detail or f"Model '{model_id}' is not available (status: {status})",
            request_id=request_id,
            context=context
        )


class ModelLoadingError(ModelError):
    """Raised when model loading fails."""
    
    def __init__(
        self,
        model_id: str,
        reason: str,
        message: str = "Model loading failed",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["reason"] = reason
        
        super().__init__(
            model_id=model_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="MODEL_LOADING_FAILED",
            message=message,
            detail=detail or f"Failed to load model '{model_id}': {reason}",
            request_id=request_id,
            context=context
        )


class ModelUnloadingError(ModelError):
    """Raised when model unloading fails."""
    
    def __init__(
        self,
        model_id: str,
        reason: str,
        message: str = "Model unloading failed",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["reason"] = reason
        
        super().__init__(
            model_id=model_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="MODEL_UNLOADING_FAILED",
            message=message,
            detail=detail or f"Failed to unload model '{model_id}': {reason}",
            request_id=request_id,
            context=context
        )


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""
    
    def __init__(
        self,
        model_id: str,
        reason: str,
        message: str = "Model inference failed",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["reason"] = reason
        
        super().__init__(
            model_id=model_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="MODEL_INFERENCE_FAILED",
            message=message,
            detail=detail or f"Inference with model '{model_id}' failed: {reason}",
            request_id=request_id,
            context=context
        )


class ModelTimeoutError(ModelError):
    """Raised when model inference times out."""
    
    def __init__(
        self,
        model_id: str,
        timeout_seconds: int,
        message: str = "Model inference timeout",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["timeout_seconds"] = timeout_seconds
        
        super().__init__(
            model_id=model_id,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error_code="MODEL_TIMEOUT",
            message=message,
            detail=detail or f"Inference with model '{model_id}' timed out after {timeout_seconds}s",
            request_id=request_id,
            context=context
        )


class ModelMemoryError(ModelError):
    """Raised when there's insufficient memory to load model."""
    
    def __init__(
        self,
        model_id: str,
        required_mb: float,
        available_mb: float,
        message: str = "Insufficient memory",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["required_mb"] = required_mb
        context["available_mb"] = available_mb
        
        super().__init__(
            model_id=model_id,
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            error_code="INSUFFICIENT_MEMORY",
            message=message,
            detail=detail or f"Insufficient memory to load model '{model_id}'. Required: {required_mb}MB, Available: {available_mb}MB",
            request_id=request_id,
            context=context
        )


class QuantizationError(ModelError):
    """Raised when model quantization fails."""
    
    def __init__(
        self,
        model_id: str,
        quantization: str,
        reason: str,
        message: str = "Quantization failed",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["quantization"] = quantization
        context["reason"] = reason
        
        super().__init__(
            model_id=model_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="QUANTIZATION_FAILED",
            message=message,
            detail=detail or f"Failed to quantize model '{model_id}' to {quantization}: {reason}",
            request_id=request_id,
            context=context
        )


# ============================================================================
# REQUEST VALIDATION EXCEPTIONS
# ============================================================================

class ValidationError(LLMGatewayException):
    """Raised when request validation fails."""
    
    def __init__(
        self,
        errors: Union[str, List[Dict[str, Any]]],
        message: str = "Validation error",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        if isinstance(errors, str):
            errors = [{"message": errors}]
        
        context = context or {}
        context["validation_errors"] = errors
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            message=message,
            detail=detail or "Request validation failed",
            request_id=request_id,
            context=context
        )


class InvalidPromptError(ValidationError):
    """Raised when prompt is invalid."""
    
    def __init__(
        self,
        reason: str,
        message: str = "Invalid prompt",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            errors=[{"field": "prompt", "message": reason}],
            message=message,
            detail=detail or f"Invalid prompt: {reason}",
            request_id=request_id,
            context=context
        )
        self.error_code = "INVALID_PROMPT"


class PromptTooLongError(ValidationError):
    """Raised when prompt exceeds maximum length."""
    
    def __init__(
        self,
        max_length: int,
        current_length: int,
        message: str = "Prompt too long",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["max_length"] = max_length
        context["current_length"] = current_length
        
        super().__init__(
            errors=[{
                "field": "prompt",
                "message": f"Prompt exceeds maximum length of {max_length} characters (current: {current_length})"
            }],
            message=message,
            detail=detail or f"Prompt length {current_length} exceeds maximum {max_length}",
            request_id=request_id,
            context=context
        )
        self.error_code = "PROMPT_TOO_LONG"


class InvalidParameterError(ValidationError):
    """Raised when a request parameter is invalid."""
    
    def __init__(
        self,
        parameter: str,
        reason: str,
        message: str = "Invalid parameter",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["parameter"] = parameter
        
        super().__init__(
            errors=[{"field": parameter, "message": reason}],
            message=message,
            detail=detail or f"Invalid parameter '{parameter}': {reason}",
            request_id=request_id,
            context=context
        )
        self.error_code = "INVALID_PARAMETER"


# ============================================================================
# CACHE EXCEPTIONS
# ============================================================================

class CacheError(LLMGatewayException):
    """Base class for cache-related exceptions."""
    
    def __init__(
        self,
        operation: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "CACHE_ERROR",
        message: str = "Cache operation failed",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["operation"] = operation
        
        super().__init__(
            status_code=status_code,
            error_code=error_code,
            message=message,
            detail=detail or f"Cache operation '{operation}' failed",
            request_id=request_id,
            context=context
        )


class CacheConnectionError(CacheError):
    """Raised when unable to connect to cache."""
    
    def __init__(
        self,
        host: str,
        port: int,
        message: str = "Cache connection failed",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["host"] = host
        context["port"] = port
        
        super().__init__(
            operation="connect",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="CACHE_CONNECTION_FAILED",
            message=message,
            detail=detail or f"Failed to connect to cache at {host}:{port}",
            request_id=request_id,
            context=context
        )


# ============================================================================
# DATABASE EXCEPTIONS
# ============================================================================

class DatabaseError(LLMGatewayException):
    """Base class for database-related exceptions."""
    
    def __init__(
        self,
        operation: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "DATABASE_ERROR",
        message: str = "Database operation failed",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["operation"] = operation
        
        super().__init__(
            status_code=status_code,
            error_code=error_code,
            message=message,
            detail=detail or f"Database operation '{operation}' failed",
            request_id=request_id,
            context=context
        )


class DatabaseConnectionError(DatabaseError):
    """Raised when unable to connect to database."""
    
    def __init__(
        self,
        message: str = "Database connection failed",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            operation="connect",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="DATABASE_CONNECTION_FAILED",
            message=message,
            detail=detail or "Failed to connect to database",
            request_id=request_id,
            context=context
        )


class IntegrityError(DatabaseError):
    """Raised when database integrity constraint is violated."""
    
    def __init__(
        self,
        constraint: str,
        message: str = "Integrity constraint violation",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["constraint"] = constraint
        
        super().__init__(
            operation="insert/update",
            status_code=status.HTTP_409_CONFLICT,
            error_code="INTEGRITY_ERROR",
            message=message,
            detail=detail or f"Database integrity constraint '{constraint}' violated",
            request_id=request_id,
            context=context
        )


# ============================================================================
# EXTERNAL API EXCEPTIONS
# ============================================================================

class ExternalAPIError(LLMGatewayException):
    """Base class for external API exceptions."""
    
    def __init__(
        self,
        provider: str,
        status_code: int = status.HTTP_502_BAD_GATEWAY,
        error_code: str = "EXTERNAL_API_ERROR",
        message: str = "External API error",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["provider"] = provider
        
        super().__init__(
            status_code=status_code,
            error_code=error_code,
            message=message,
            detail=detail or f"External API provider '{provider}' returned an error",
            request_id=request_id,
            context=context
        )


class GrokAPIError(ExternalAPIError):
    """Raised when Grok API returns an error."""
    
    def __init__(
        self,
        status: int,
        response: str,
        message: str = "Grok API error",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["api_status"] = status
        context["api_response"] = response[:200]  # Truncate long responses
        
        super().__init__(
            provider="Grok",
            status_code=status if status >= 500 else status.HTTP_502_BAD_GATEWAY,
            error_code="GROK_API_ERROR",
            message=message,
            detail=detail or f"Grok API returned error {status}: {response[:100]}",
            request_id=request_id,
            context=context
        )


class OpenAIAPIError(ExternalAPIError):
    """Raised when OpenAI API returns an error."""
    
    def __init__(
        self,
        status: int,
        response: str,
        message: str = "OpenAI API error",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["api_status"] = status
        context["api_response"] = response[:200]
        
        super().__init__(
            provider="OpenAI",
            status_code=status if status >= 500 else status.HTTP_502_BAD_GATEWAY,
            error_code="OPENAI_API_ERROR",
            message=message,
            detail=detail or f"OpenAI API returned error {status}: {response[:100]}",
            request_id=request_id,
            context=context
        )


class AnthropicAPIError(ExternalAPIError):
    """Raised when Anthropic API returns an error."""
    
    def __init__(
        self,
        status: int,
        response: str,
        message: str = "Anthropic API error",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["api_status"] = status
        context["api_response"] = response[:200]
        
        super().__init__(
            provider="Anthropic",
            status_code=status if status >= 500 else status.HTTP_502_BAD_GATEWAY,
            error_code="ANTHROPIC_API_ERROR",
            message=message,
            detail=detail or f"Anthropic API returned error {status}: {response[:100]}",
            request_id=request_id,
            context=context
        )


class CohereAPIError(ExternalAPIError):
    """Raised when Cohere API returns an error."""
    
    def __init__(
        self,
        status: int,
        response: str,
        message: str = "Cohere API error",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["api_status"] = status
        context["api_response"] = response[:200]
        
        super().__init__(
            provider="Cohere",
            status_code=status if status >= 500 else status.HTTP_502_BAD_GATEWAY,
            error_code="COHERE_API_ERROR",
            message=message,
            detail=detail or f"Cohere API returned error {status}: {response[:100]}",
            request_id=request_id,
            context=context
        )


class APIKeyMissingError(ExternalAPIError):
    """Raised when API key is missing for external service."""
    
    def __init__(
        self,
        provider: str,
        message: str = "API key missing",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            provider=provider,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="API_KEY_MISSING",
            message=message,
            detail=detail or f"API key for {provider} is not configured",
            request_id=request_id,
            context=context
        )


class APIKeyInvalidError(ExternalAPIError):
    """Raised when API key is invalid for external service."""
    
    def __init__(
        self,
        provider: str,
        message: str = "Invalid API key",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            provider=provider,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="API_KEY_INVALID",
            message=message,
            detail=detail or f"API key for {provider} is invalid",
            request_id=request_id,
            context=context
        )


# ============================================================================
# ROUTING EXCEPTIONS
# ============================================================================

class RoutingError(LLMGatewayException):
    """Base class for routing-related exceptions."""
    
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "ROUTING_ERROR",
        message: str = "Routing error occurred",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status_code,
            error_code=error_code,
            message=message,
            detail=detail,
            request_id=request_id,
            context=context
        )


class NoSuitableModelError(RoutingError):
    """Raised when no suitable model is found for routing."""
    
    def __init__(
        self,
        criteria: Dict[str, Any],
        message: str = "No suitable model found",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["criteria"] = criteria
        
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="NO_SUITABLE_MODEL",
            message=message,
            detail=detail or "No model matches the specified routing criteria",
            request_id=request_id,
            context=context
        )


class RoutingStrategyError(RoutingError):
    """Raised when routing strategy fails."""
    
    def __init__(
        self,
        strategy: str,
        reason: str,
        message: str = "Routing strategy failed",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["strategy"] = strategy
        context["reason"] = reason
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="ROUTING_STRATEGY_FAILED",
            message=message,
            detail=detail or f"Routing strategy '{strategy}' failed: {reason}",
            request_id=request_id,
            context=context
        )


# ============================================================================
# RESOURCE EXCEPTIONS
# ============================================================================

class ResourceNotFoundError(LLMGatewayException):
    """Raised when a requested resource is not found."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        message: str = "Resource not found",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["resource_type"] = resource_type
        context["resource_id"] = resource_id
        
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="RESOURCE_NOT_FOUND",
            message=message,
            detail=detail or f"{resource_type} '{resource_id}' not found",
            request_id=request_id,
            context=context
        )


class ResourceConflictError(LLMGatewayException):
    """Raised when creating a resource that already exists."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        message: str = "Resource already exists",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["resource_type"] = resource_type
        context["resource_id"] = resource_id
        
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            error_code="RESOURCE_CONFLICT",
            message=message,
            detail=detail or f"{resource_type} '{resource_id}' already exists",
            request_id=request_id,
            context=context
        )


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================

class ConfigurationError(LLMGatewayException):
    """Raised when there's a configuration error."""
    
    def __init__(
        self,
        config_key: str,
        message: str = "Configuration error",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["config_key"] = config_key
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="CONFIGURATION_ERROR",
            message=message,
            detail=detail or f"Configuration error for '{config_key}'",
            request_id=request_id,
            context=context
        )


class ConfigurationFileNotFoundError(ConfigurationError):
    """Raised when configuration file is not found."""
    
    def __init__(
        self,
        file_path: str,
        message: str = "Configuration file not found",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["file_path"] = file_path
        
        super().__init__(
            config_key="file",
            message=message,
            detail=detail or f"Configuration file not found: {file_path}",
            request_id=request_id,
            context=context
        )
        self.error_code = "CONFIG_FILE_NOT_FOUND"


# ============================================================================
# MAINTENANCE EXCEPTIONS
# ============================================================================

class MaintenanceModeError(LLMGatewayException):
    """Raised when system is in maintenance mode."""
    
    def __init__(
        self,
        message: str = "System in maintenance mode",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        estimated_duration: Optional[int] = None
    ):
        context = context or {}
        if estimated_duration:
            context["estimated_duration_minutes"] = estimated_duration
        
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MAINTENANCE_MODE",
            message=message,
            detail=detail or "System is currently undergoing maintenance",
            request_id=request_id,
            context=context,
            headers={"Retry-After": str(estimated_duration * 60) if estimated_duration else "3600"}
        )


# ============================================================================
# CIRCUIT BREAKER EXCEPTIONS
# ============================================================================

class CircuitBreakerError(LLMGatewayException):
    """Raised when circuit breaker is open."""
    
    def __init__(
        self,
        service: str,
        message: str = "Circuit breaker open",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        retry_after: Optional[int] = None
    ):
        context = context or {}
        context["service"] = service
        if retry_after:
            context["retry_after_seconds"] = retry_after
        
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="CIRCUIT_BREAKER_OPEN",
            message=message,
            detail=detail or f"Circuit breaker is open for service '{service}'",
            request_id=request_id,
            context=context,
            headers={"Retry-After": str(retry_after) if retry_after else "60"}
        )


# ============================================================================
# FILE PROCESSING EXCEPTIONS
# ============================================================================

class FileProcessingError(LLMGatewayException):
    """Base class for file processing exceptions."""
    
    def __init__(
        self,
        filename: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "FILE_PROCESSING_ERROR",
        message: str = "File processing failed",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["filename"] = filename
        
        super().__init__(
            status_code=status_code,
            error_code=error_code,
            message=message,
            detail=detail,
            request_id=request_id,
            context=context
        )


class FileTooLargeError(FileProcessingError):
    """Raised when file exceeds size limit."""
    
    def __init__(
        self,
        filename: str,
        size_mb: float,
        limit_mb: float,
        message: str = "File too large",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["size_mb"] = size_mb
        context["limit_mb"] = limit_mb
        
        super().__init__(
            filename=filename,
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            error_code="FILE_TOO_LARGE",
            message=message,
            detail=detail or f"File size {size_mb}MB exceeds limit of {limit_mb}MB",
            request_id=request_id,
            context=context
        )


class UnsupportedFileFormatError(FileProcessingError):
    """Raised when file format is not supported."""
    
    def __init__(
        self,
        filename: str,
        format: str,
        supported_formats: List[str],
        message: str = "Unsupported file format",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["format"] = format
        context["supported_formats"] = supported_formats
        
        super().__init__(
            filename=filename,
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            error_code="UNSUPPORTED_FILE_FORMAT",
            message=message,
            detail=detail or f"File format '{format}' is not supported. Supported formats: {', '.join(supported_formats)}",
            request_id=request_id,
            context=context
        )


# ============================================================================
# BATCH PROCESSING EXCEPTIONS
# ============================================================================

class BatchJobError(LLMGatewayException):
    """Raised when batch job processing fails."""
    
    def __init__(
        self,
        job_id: str,
        reason: str,
        message: str = "Batch job failed",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["job_id"] = job_id
        context["reason"] = reason
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="BATCH_JOB_FAILED",
            message=message,
            detail=detail or f"Batch job '{job_id}' failed: {reason}",
            request_id=request_id,
            context=context
        )


# ============================================================================
# ERROR CODE MAPPING
# ============================================================================

ERROR_CODE_MAPPING = {
    # Authentication (401)
    "AUTHENTICATION_FAILED": AuthenticationError,
    "INVALID_API_KEY": InvalidAPIKeyError,
    "EXPIRED_API_KEY": ExpiredAPIKeyError,
    "INVALID_TOKEN": InvalidTokenError,
    "EXPIRED_TOKEN": ExpiredTokenError,
    
    # Authorization (403)
    "ADMIN_REQUIRED": AdminPermissionError,
    "INSUFFICIENT_PERMISSIONS": InsufficientPermissionsError,
    
    # Rate Limiting (429)
    "RATE_LIMIT_EXCEEDED": RateLimitExceededError,
    "CONCURRENT_LIMIT_EXCEEDED": ConcurrentRequestLimitError,
    
    # Models (404, 503, 504, 507)
    "MODEL_NOT_FOUND": ModelNotFoundError,
    "MODEL_NOT_AVAILABLE": ModelNotAvailableError,
    "MODEL_LOADING_FAILED": ModelLoadingError,
    "MODEL_UNLOADING_FAILED": ModelUnloadingError,
    "MODEL_INFERENCE_FAILED": ModelInferenceError,
    "MODEL_TIMEOUT": ModelTimeoutError,
    "INSUFFICIENT_MEMORY": ModelMemoryError,
    "QUANTIZATION_FAILED": QuantizationError,
    
    # Validation (422)
    "VALIDATION_ERROR": ValidationError,
    "INVALID_PROMPT": InvalidPromptError,
    "PROMPT_TOO_LONG": PromptTooLongError,
    "INVALID_PARAMETER": InvalidParameterError,
    
    # Cache (503)
    "CACHE_CONNECTION_FAILED": CacheConnectionError,
    
    # Database (409, 503)
    "DATABASE_CONNECTION_FAILED": DatabaseConnectionError,
    "INTEGRITY_ERROR": IntegrityError,
    
    # External APIs (401, 502)
    "GROK_API_ERROR": GrokAPIError,
    "OPENAI_API_ERROR": OpenAIAPIError,
    "ANTHROPIC_API_ERROR": AnthropicAPIError,
    "COHERE_API_ERROR": CohereAPIError,
    "API_KEY_MISSING": APIKeyMissingError,
    "API_KEY_INVALID": APIKeyInvalidError,
    
    # Routing (503)
    "NO_SUITABLE_MODEL": NoSuitableModelError,
    "ROUTING_STRATEGY_FAILED": RoutingStrategyError,
    
    # Resources (404, 409)
    "RESOURCE_NOT_FOUND": ResourceNotFoundError,
    "RESOURCE_CONFLICT": ResourceConflictError,
    
    # Configuration (500)
    "CONFIGURATION_ERROR": ConfigurationError,
    "CONFIG_FILE_NOT_FOUND": ConfigurationFileNotFoundError,
    
    # Maintenance (503)
    "MAINTENANCE_MODE": MaintenanceModeError,
    
    # Circuit Breaker (503)
    "CIRCUIT_BREAKER_OPEN": CircuitBreakerError,
    
    # File Processing (413, 415)
    "FILE_TOO_LARGE": FileTooLargeError,
    "UNSUPPORTED_FILE_FORMAT": UnsupportedFileFormatError,
    
    # Batch Processing (500)
    "BATCH_JOB_FAILED": BatchJobError,
}


def get_exception_by_error_code(error_code: str, **kwargs) -> LLMGatewayException:
    """
    Factory method to create exception from error code.
    
    Args:
        error_code: String error code
        **kwargs: Exception constructor arguments
    
    Returns:
        Instantiated exception object
    
    Raises:
        KeyError: If error_code is not mapped
    """
    if error_code not in ERROR_CODE_MAPPING:
        raise KeyError(f"Unknown error code: {error_code}")
    
    exception_class = ERROR_CODE_MAPPING[error_code]
    return exception_class(**kwargs)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Base
    "LLMGatewayException",
    
    # Authentication & Authorization
    "AuthenticationError",
    "InvalidAPIKeyError",
    "ExpiredAPIKeyError",
    "InvalidTokenError",
    "ExpiredTokenError",
    "AdminPermissionError",
    "InsufficientPermissionsError",
    
    # Rate Limiting
    "RateLimitExceededError",
    "ConcurrentRequestLimitError",
    
    # Models
    "ModelError",
    "ModelNotFoundError",
    "ModelNotAvailableError",
    "ModelLoadingError",
    "ModelUnloadingError",
    "ModelInferenceError",
    "ModelTimeoutError",
    "ModelMemoryError",
    "QuantizationError",
    
    # Validation
    "ValidationError",
    "InvalidPromptError",
    "PromptTooLongError",
    "InvalidParameterError",
    
    # Cache
    "CacheError",
    "CacheConnectionError",
    
    # Database
    "DatabaseError",
    "DatabaseConnectionError",
    "IntegrityError",
    
    # External APIs
    "ExternalAPIError",
    "GrokAPIError",
    "OpenAIAPIError",
    "AnthropicAPIError",
    "CohereAPIError",
    "APIKeyMissingError",
    "APIKeyInvalidError",
    
    # Routing
    "RoutingError",
    "NoSuitableModelError",
    "RoutingStrategyError",
    
    # Resources
    "ResourceNotFoundError",
    "ResourceConflictError",
    
    # Configuration
    "ConfigurationError",
    "ConfigurationFileNotFoundError",
    
    # Maintenance
    "MaintenanceModeError",
    
    # Circuit Breaker
    "CircuitBreakerError",
    
    # File Processing
    "FileProcessingError",
    "FileTooLargeError",
    "UnsupportedFileFormatError",
    
    # Batch Processing
    "BatchJobError",
    
    # Factory
    "get_exception_by_error_code",
    "ERROR_CODE_MAPPING",
]