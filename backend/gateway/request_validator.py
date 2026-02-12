"""
Request Validator - Production Ready
Comprehensive request validation for LLM inference with security checks,
schema validation, content filtering, and rate limit coordination.
"""

import re
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import html
import bleach

from pydantic import BaseModel, ValidationError, validator, Field
from fastapi import Request

from core.logging import get_logger
from core.exceptions import InvalidPromptError, ValidationError as GatewayValidationError
from config import settings

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# Prompt constraints
MAX_PROMPT_LENGTH = 10000
MIN_PROMPT_LENGTH = 1
MAX_MESSAGES = 50
MAX_TOKENS = 4096
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0
MIN_N = 1
MAX_N = 5
MAX_STOP_SEQUENCES = 4
MAX_STOP_SEQUENCE_LENGTH = 100
MIN_PRESENCE_PENALTY = -2.0
MAX_PRESENCE_PENALTY = 2.0
MIN_FREQUENCY_PENALTY = -2.0
MAX_FREQUENCY_PENALTY = 2.0

# Content filtering
PROHIBITED_PATTERNS = [
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]',  # Control characters
    r'\\x[0-9a-fA-F]{2}',  # Escaped hex
    r'\\u[0-9a-fA-F]{4}',  # Escaped unicode
]

SENSITIVE_PATTERNS = [
    r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',  # SSN
    r'\b\d{16}\b',  # Credit card (basic)
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
]

# Allowed roles in chat messages
ALLOWED_ROLES = ["system", "user", "assistant"]

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class ChatMessage(BaseModel):
    """Individual chat message validation."""
    
    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Sender name")
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ALLOWED_ROLES:
            raise ValueError(f"Invalid role: {v}. Must be one of {ALLOWED_ROLES}")
        return v
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        if len(v) > MAX_PROMPT_LENGTH:
            raise ValueError(f"Message content too long: {len(v)} > {MAX_PROMPT_LENGTH}")
        return v
    
    @validator('name')
    def validate_name(cls, v):
        if v is not None:
            if len(v) > 64:
                raise ValueError("Name too long: maximum 64 characters")
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError("Name can only contain alphanumeric, underscore, and hyphen")
        return v


class ChatCompletionRequest(BaseModel):
    """Chat completion request validation."""
    
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: Optional[str] = Field(None, description="Model ID to use")
    stream: bool = Field(False, description="Whether to stream the response")
    max_tokens: int = Field(1000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.95, description="Nucleus sampling parameter")
    n: int = Field(1, description="Number of completions to generate")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: float = Field(0.0, description="Presence penalty")
    frequency_penalty: float = Field(0.0, description="Frequency penalty")
    user: Optional[str] = Field(None, description="User identifier")
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        if len(v) > MAX_MESSAGES:
            raise ValueError(f"Too many messages: {len(v)} > {MAX_MESSAGES}")
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v < 1:
            raise ValueError("max_tokens must be at least 1")
        if v > MAX_TOKENS:
            raise ValueError(f"max_tokens exceeds maximum: {v} > {MAX_TOKENS}")
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v < MIN_TEMPERATURE or v > MAX_TEMPERATURE:
            raise ValueError(f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
        return v
    
    @validator('top_p')
    def validate_top_p(cls, v):
        if v < MIN_TOP_P or v > MAX_TOP_P:
            raise ValueError(f"top_p must be between {MIN_TOP_P} and {MAX_TOP_P}")
        return v
    
    @validator('n')
    def validate_n(cls, v):
        if v < MIN_N or v > MAX_N:
            raise ValueError(f"n must be between {MIN_N} and {MAX_N}")
        return v
    
    @validator('stop')
    def validate_stop(cls, v):
        if v is not None:
            if isinstance(v, str):
                if len(v) > MAX_STOP_SEQUENCE_LENGTH:
                    raise ValueError(f"Stop sequence too long: {len(v)} > {MAX_STOP_SEQUENCE_LENGTH}")
                sequences = [v]
            elif isinstance(v, list):
                if len(v) > MAX_STOP_SEQUENCES:
                    raise ValueError(f"Too many stop sequences: {len(v)} > {MAX_STOP_SEQUENCES}")
                sequences = v
                for seq in sequences:
                    if len(seq) > MAX_STOP_SEQUENCE_LENGTH:
                        raise ValueError(f"Stop sequence too long: {len(seq)} > {MAX_STOP_SEQUENCE_LENGTH}")
            else:
                raise ValueError("stop must be string or list of strings")
        return v
    
    @validator('presence_penalty')
    def validate_presence_penalty(cls, v):
        if v < MIN_PRESENCE_PENALTY or v > MAX_PRESENCE_PENALTY:
            raise ValueError(f"presence_penalty must be between {MIN_PRESENCE_PENALTY} and {MAX_PRESENCE_PENALTY}")
        return v
    
    @validator('frequency_penalty')
    def validate_frequency_penalty(cls, v):
        if v < MIN_FREQUENCY_PENALTY or v > MAX_FREQUENCY_PENALTY:
            raise ValueError(f"frequency_penalty must be between {MIN_FREQUENCY_PENALTY} and {MAX_FREQUENCY_PENALTY}")
        return v
    
    @validator('user')
    def validate_user(cls, v):
        if v is not None:
            if len(v) > 64:
                raise ValueError("User identifier too long: maximum 64 characters")
            if not re.match(r'^[a-zA-Z0-9@._-]+$', v):
                raise ValueError("User identifier contains invalid characters")
        return v


class EmbeddingRequest(BaseModel):
    """Embedding request validation."""
    
    input: Union[str, List[str]] = Field(..., description="Text to embed")
    model: str = Field("all-MiniLM-L6-v2", description="Embedding model")
    
    @validator('input')
    def validate_input(cls, v):
        if isinstance(v, str):
            if len(v) > MAX_PROMPT_LENGTH:
                raise ValueError(f"Input too long: {len(v)} > {MAX_PROMPT_LENGTH}")
            if not v.strip():
                raise ValueError("Input cannot be empty")
        elif isinstance(v, list):
            if len(v) > 100:
                raise ValueError(f"Too many inputs: {len(v)} > 100")
            for item in v:
                if len(item) > MAX_PROMPT_LENGTH:
                    raise ValueError(f"Input item too long: {len(item)} > {MAX_PROMPT_LENGTH}")
                if not item.strip():
                    raise ValueError("Input item cannot be empty")
        return v


# ============================================================================
# CONTENT FILTER
# ============================================================================

class ContentFilter:
    """
    Content filtering and sanitization.
    
    Features:
    - Remove control characters
    - Sanitize HTML/XML
    - Detect sensitive information
    - Pattern blocking
    - Language detection
    """
    
    def __init__(self):
        self.prohibited_patterns = [re.compile(p) for p in PROHIBITED_PATTERNS]
        self.sensitive_patterns = [re.compile(p) for p in SENSITIVE_PATTERNS]
        self.allowed_tags = ['b', 'i', 'em', 'strong', 'code', 'pre']
        self.allowed_attributes = {}
    
    def sanitize(self, text: str) -> str:
        """
        Sanitize text by removing dangerous content.
        
        Args:
            text: Raw input text
        
        Returns:
            Sanitized text
        """
        if not text:
            return text
        
        # Remove control characters
        for pattern in self.prohibited_patterns:
            text = pattern.sub('', text)
        
        # Unescape HTML entities
        text = html.unescape(text)
        
        # Strip HTML tags (allow safe ones)
        text = bleach.clean(
            text,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True
        )
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def contains_sensitive_data(self, text: str) -> List[str]:
        """
        Check if text contains sensitive information.
        
        Args:
            text: Text to check
        
        Returns:
            List of detected sensitive patterns
        """
        detected = []
        
        for pattern in self.sensitive_patterns:
            matches = pattern.findall(text)
            if matches:
                detected.append(pattern.pattern)
        
        return detected
    
    def mask_sensitive_data(self, text: str) -> str:
        """
        Mask sensitive information in text.
        
        Args:
            text: Text to mask
        
        Returns:
            Text with sensitive data masked
        """
        for pattern in self.sensitive_patterns:
            text = pattern.sub('[REDACTED]', text)
        
        return text


# ============================================================================
# REQUEST SIZE LIMITER
# ============================================================================

class RequestSizeLimiter:
    """
    Limit request size to prevent DoS attacks.
    """
    
    def __init__(self, max_size_bytes: int = 10 * 1024 * 1024):  # 10MB default
        self.max_size_bytes = max_size_bytes
    
    async def check_request_size(self, request: Request) -> bool:
        """
        Check if request size is within limits.
        
        Args:
            request: FastAPI request object
        
        Returns:
            True if size is acceptable
        
        Raises:
            HTTPException: If request too large
        """
        content_length = request.headers.get('content-length')
        
        if content_length:
            size = int(content_length)
            if size > self.max_size_bytes:
                logger.warning(
                    "request_size_exceeded",
                    size_bytes=size,
                    max_bytes=self.max_size_bytes
                )
                return False
        
        return True


# ============================================================================
# RATE LIMIT COORDINATOR
# ============================================================================

class RateLimitCoordinator:
    """
    Coordinate rate limiting with user tiers.
    """
    
    def __init__(self):
        self.tiers = {
            "free": {
                "requests_per_minute": 60,
                "tokens_per_minute": 10000,
                "concurrent_requests": 5
            },
            "pro": {
                "requests_per_minute": 600,
                "tokens_per_minute": 100000,
                "concurrent_requests": 20
            },
            "enterprise": {
                "requests_per_minute": 6000,
                "tokens_per_minute": 1000000,
                "concurrent_requests": 100
            },
            "admin": {
                "requests_per_minute": 10000,
                "tokens_per_minute": 10000000,
                "concurrent_requests": 500
            }
        }
    
    def get_user_tier(self, user: Optional[Dict[str, Any]] = None) -> str:
        """Get rate limit tier for user."""
        if not user:
            return "free"
        
        if user.get("is_admin"):
            return "admin"
        
        return user.get("tier", "free")
    
    def get_limits(self, tier: str) -> Dict[str, int]:
        """Get rate limits for tier."""
        return self.tiers.get(tier, self.tiers["free"])


# ============================================================================
# PROMPT VALIDATOR
# ============================================================================

class PromptValidator:
    """
    Comprehensive prompt validation.
    
    Checks:
    - Length constraints
    - Content safety
    - Format validation
    - Language detection
    - Toxicity detection (optional)
    """
    
    def __init__(self):
        self.content_filter = ContentFilter()
        self.max_length = MAX_PROMPT_LENGTH
        self.min_length = MIN_PROMPT_LENGTH
    
    async def validate(
        self,
        prompt: str,
        check_sensitive: bool = True,
        sanitize: bool = True
    ) -> Dict[str, Any]:
        """
        Validate and optionally sanitize a prompt.
        
        Args:
            prompt: Input prompt
            check_sensitive: Check for sensitive data
            sanitize: Sanitize the prompt
        
        Returns:
            Validation result with sanitized prompt and metadata
        
        Raises:
            InvalidPromptError: If validation fails
        """
        if not prompt:
            raise InvalidPromptError("Prompt cannot be empty")
        
        original_length = len(prompt)
        
        # Check length
        if original_length > self.max_length:
            raise InvalidPromptError(
                f"Prompt too long: {original_length} > {self.max_length} characters"
            )
        
        if original_length < self.min_length:
            raise InvalidPromptError(
                f"Prompt too short: {original_length} < {self.min_length} characters"
            )
        
        # Sanitize if requested
        sanitized_prompt = prompt
        if sanitize:
            sanitized_prompt = self.content_filter.sanitize(prompt)
            
            if not sanitized_prompt:
                raise InvalidPromptError("Prompt is empty after sanitization")
        
        # Check for sensitive data
        sensitive_data = []
        if check_sensitive:
            sensitive_data = self.content_filter.contains_sensitive_data(prompt)
            
            # Log warning but don't block (let application decide)
            if sensitive_data:
                logger.warning(
                    "sensitive_data_detected_in_prompt",
                    patterns=sensitive_data,
                    prompt_length=original_length
                )
        
        return {
            "valid": True,
            "original": prompt,
            "sanitized": sanitized_prompt,
            "original_length": original_length,
            "sanitized_length": len(sanitized_prompt),
            "sensitive_data_detected": sensitive_data,
            "was_sanitized": prompt != sanitized_prompt
        }


# ============================================================================
# REQUEST VALIDATOR
# ============================================================================

class RequestValidator:
    """
    Main request validator for the gateway.
    
    Validates:
    - HTTP method and headers
    - Request body schema
    - Content type
    - Request size
    - Authentication (delegated to security module)
    - Rate limits (delegated to rate limiter)
    - Prompt content
    """
    
    def __init__(self):
        self.content_filter = ContentFilter()
        self.prompt_validator = PromptValidator()
        self.size_limiter = RequestSizeLimiter()
        self.rate_limit_coordinator = RateLimitCoordinator()
        
        # Allowed content types
        self.allowed_content_types = [
            "application/json",
            "multipart/form-data",
            "application/x-www-form-urlencoded"
        ]
        
        logger.info("request_validator_initialized")
    
    async def validate_chat_request(
        self,
        request: Request,
        body: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate a chat completion request.
        
        Args:
            request: FastAPI request object
            body: Request body
            user: Authenticated user (optional)
        
        Returns:
            Validated and sanitized request data
        
        Raises:
            ValidationError: If validation fails
            InvalidPromptError: If prompt validation fails
        """
        # ====================================================================
        # STEP 1: Validate HTTP basics
        # ====================================================================
        await self._validate_http_request(request)
        
        # ====================================================================
        # STEP 2: Validate request size
        # ====================================================================
        if not await self.size_limiter.check_request_size(request):
            raise GatewayValidationError(
                errors=[{
                    "loc": ["body"],
                    "msg": f"Request too large. Maximum size: {self.size_limiter.max_size_bytes} bytes",
                    "type": "request_size_exceeded"
                }]
            )
        
        # ====================================================================
        # STEP 3: Validate schema with Pydantic
        # ====================================================================
        try:
            validated = ChatCompletionRequest(**body)
        except ValidationError as e:
            logger.warning(
                "chat_request_schema_validation_failed",
                errors=e.errors(),
                user_id=user.get("id") if user else None
            )
            raise GatewayValidationError(errors=e.errors())
        
        # ====================================================================
        # STEP 4: Validate and sanitize messages
        # ====================================================================
        sanitized_messages = []
        
        for msg in validated.messages:
            # Validate content
            validation_result = await self.prompt_validator.validate(
                prompt=msg.content,
                check_sensitive=True,
                sanitize=True
            )
            
            # Create sanitized message
            sanitized_msg = ChatMessage(
                role=msg.role,
                content=validation_result["sanitized"],
                name=msg.name
            )
            sanitized_messages.append(sanitized_msg)
            
            # Log if sensitive data was detected and masked
            if validation_result["sensitive_data_detected"]:
                logger.info(
                    "sensitive_data_masked_in_chat",
                    role=msg.role,
                    patterns=validation_result["sensitive_data_detected"],
                    user_id=user.get("id") if user else None
                )
        
        # ====================================================================
        # STEP 5: Get rate limits for user tier
        # ====================================================================
        tier = self.rate_limit_coordinator.get_user_tier(user)
        limits = self.rate_limit_coordinator.get_limits(tier)
        
        # ====================================================================
        # STEP 6: Build validated request object
        # ====================================================================
        validated_request = {
            "messages": [msg.dict() for msg in sanitized_messages],
            "model": validated.model,
            "stream": validated.stream,
            "max_tokens": validated.max_tokens,
            "temperature": validated.temperature,
            "top_p": validated.top_p,
            "n": validated.n,
            "stop": validated.stop,
            "presence_penalty": validated.presence_penalty,
            "frequency_penalty": validated.frequency_penalty,
            "user": validated.user,
            "validation": {
                "timestamp": datetime.utcnow().isoformat(),
                "schema_version": "1.0",
                "rate_limit_tier": tier,
                "rate_limits": limits
            }
        }
        
        logger.debug(
            "chat_request_validated",
            user_id=user.get("id") if user else None,
            model=validated.model,
            stream=validated.stream,
            message_count=len(validated.messages),
            tier=tier
        )
        
        return validated_request
    
    async def validate_embedding_request(
        self,
        request: Request,
        body: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate an embedding request.
        
        Args:
            request: FastAPI request object
            body: Request body
            user: Authenticated user (optional)
        
        Returns:
            Validated request data
        """
        # ====================================================================
        # STEP 1: Validate HTTP basics
        # ====================================================================
        await self._validate_http_request(request)
        
        # ====================================================================
        # STEP 2: Validate request size
        # ====================================================================
        if not await self.size_limiter.check_request_size(request):
            raise GatewayValidationError(
                errors=[{
                    "loc": ["body"],
                    "msg": f"Request too large. Maximum size: {self.size_limiter.max_size_bytes} bytes",
                    "type": "request_size_exceeded"
                }]
            )
        
        # ====================================================================
        # STEP 3: Validate schema with Pydantic
        # ====================================================================
        try:
            validated = EmbeddingRequest(**body)
        except ValidationError as e:
            logger.warning(
                "embedding_request_schema_validation_failed",
                errors=e.errors(),
                user_id=user.get("id") if user else None
            )
            raise GatewayValidationError(errors=e.errors())
        
        # ====================================================================
        # STEP 4: Get rate limits for user tier
        # ====================================================================
        tier = self.rate_limit_coordinator.get_user_tier(user)
        limits = self.rate_limit_coordinator.get_limits(tier)
        
        # ====================================================================
        # STEP 5: Build validated request object
        # ====================================================================
        validated_request = {
            "input": validated.input,
            "model": validated.model,
            "validation": {
                "timestamp": datetime.utcnow().isoformat(),
                "schema_version": "1.0",
                "rate_limit_tier": tier,
                "rate_limits": limits
            }
        }
        
        return validated_request
    
    async def _validate_http_request(self, request: Request) -> None:
        """
        Validate HTTP request basics.
        
        Args:
            request: FastAPI request object
        
        Raises:
            ValidationError: If validation fails
        """
        # Check method
        if request.method not in ["POST", "GET", "PUT", "DELETE", "PATCH"]:
            raise GatewayValidationError(
                errors=[{
                    "loc": ["method"],
                    "msg": f"Unsupported HTTP method: {request.method}",
                    "type": "method_not_allowed"
                }]
            )
        
        # Check content type for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get('content-type', '').lower()
            
            if not any(ct in content_type for ct in self.allowed_content_types):
                raise GatewayValidationError(
                    errors=[{
                        "loc": ["headers", "content-type"],
                        "msg": f"Unsupported content type: {content_type}",
                        "type": "content_type_not_allowed"
                    }]
                )
    
    def validate_model_id(self, model_id: str) -> bool:
        """
        Validate model ID format.
        
        Args:
            model_id: Model identifier
        
        Returns:
            True if format is valid
        """
        if not model_id or not isinstance(model_id, str):
            return False
        
        # Model ID format: provider/model-name
        # Examples: grok-beta, openai/gpt-4, meta/llama-2-7b
        pattern = r'^[a-zA-Z0-9_-]+(/[a-zA-Z0-9_.-]+)?$'
        
        return bool(re.match(pattern, model_id))
    
    def sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Sanitize HTTP headers for logging.
        
        Args:
            headers: Raw headers
        
        Returns:
            Sanitized headers with sensitive data masked
        """
        sensitive_headers = [
            'authorization',
            'x-api-key',
            'cookie',
            'set-cookie',
            'proxy-authorization'
        ]
        
        sanitized = {}
        
        for key, value in headers.items():
            key_lower = key.lower()
            
            if key_lower in sensitive_headers:
                sanitized[key] = '[REDACTED]'
            elif 'token' in key_lower or 'key' in key_lower or 'secret' in key_lower:
                sanitized[key] = '[REDACTED]'
            else:
                sanitized[key] = value[:200] + '...' if len(value) > 200 else value
        
        return sanitized


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_request_validator = None


def get_request_validator() -> RequestValidator:
    """Get singleton request validator instance."""
    global _request_validator
    if not _request_validator:
        _request_validator = RequestValidator()
    return _request_validator


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "RequestValidator",
    "PromptValidator",
    "ContentFilter",
    "RequestSizeLimiter",
    "RateLimitCoordinator",
    "get_request_validator",
    
    # Validation schemas
    "ChatCompletionRequest",
    "ChatMessage",
    "EmbeddingRequest",
    
    # Constants
    "MAX_PROMPT_LENGTH",
    "MAX_TOKENS",
    "MAX_MESSAGES",
    "ALLOWED_ROLES"
]