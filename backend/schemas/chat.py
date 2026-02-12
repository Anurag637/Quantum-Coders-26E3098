"""
Chat Schemas - Production Ready
Pydantic models for chat completion requests and responses with comprehensive
validation, type safety, and OpenAI API compatibility.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
import re

from core.exceptions import ValidationError

# ============================================================================
# CHAT MESSAGE SCHEMAS
# ============================================================================

class ChatMessage(BaseModel):
    """
    Individual chat message in a conversation.
    
    OpenAI-compatible format with role and content.
    Supports system, user, and assistant messages.
    """
    
    role: str = Field(
        ...,
        description="Role of the message sender",
        example="user"
    )
    content: str = Field(
        ...,
        description="Content of the message",
        example="What is the capital of France?"
    )
    name: Optional[str] = Field(
        None,
        description="Name of the sender (for multi-user conversations)",
        example="John"
    )
    
    @validator('role')
    def validate_role(cls, v):
        """Validate message role."""
        allowed_roles = ['system', 'user', 'assistant']
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v
    
    @validator('content')
    def validate_content(cls, v):
        """Validate message content."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        if len(v) > 100000:
            raise ValueError("Message content too long (max 100000 characters)")
        return v.strip()
    
    @validator('name')
    def validate_name(cls, v):
        """Validate sender name."""
        if v is not None:
            if len(v) > 64:
                raise ValueError("Name too long (max 64 characters)")
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError("Name can only contain alphanumeric, underscore, and hyphen")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, how are you?",
                "name": "john_doe"
            }
        }


class ChatMessageList(BaseModel):
    """
    List of chat messages with validation.
    Used internally for batch operations.
    """
    
    messages: List[ChatMessage] = Field(
        ...,
        description="List of chat messages",
        min_items=1,
        max_items=100
    )
    
    @validator('messages')
    def validate_message_sequence(cls, v):
        """Validate message sequence."""
        if not v:
            raise ValueError("Messages list cannot be empty")
        
        # Check for alternating roles
        for i in range(1, len(v)):
            if v[i].role == v[i-1].role:
                # Allow consecutive assistant messages, but not user-user or system-system
                if v[i].role != 'assistant':
                    raise ValueError(f"Consecutive {v[i].role} messages are not allowed")
        
        # First message should not be assistant
        if v[0].role == 'assistant':
            raise ValueError("First message cannot be from assistant")
        
        return v


# ============================================================================
# CHAT COMPLETION REQUEST SCHEMAS
# ============================================================================

class ChatCompletionRequest(BaseModel):
    """
    Chat completion request schema.
    
    OpenAI-compatible format with support for streaming,
    temperature control, token limits, and penalties.
    """
    
    messages: List[ChatMessage] = Field(
        ...,
        description="List of messages in the conversation",
        min_items=1,
        max_items=100,
        example=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    
    model: Optional[str] = Field(
        None,
        description="ID of the model to use. If not specified, auto-routing will be used",
        example="gpt-3.5-turbo"
    )
    
    stream: bool = Field(
        False,
        description="Whether to stream the response token by token",
        example=False
    )
    
    max_tokens: Optional[int] = Field(
        1000,
        description="Maximum number of tokens to generate",
        ge=1,
        le=4096,
        example=500
    )
    
    temperature: float = Field(
        0.7,
        description="Sampling temperature (0-2). Higher values make output more random",
        ge=0.0,
        le=2.0,
        example=0.8
    )
    
    top_p: float = Field(
        0.95,
        description="Nucleus sampling: only consider tokens with top_p probability",
        ge=0.0,
        le=1.0,
        example=0.9
    )
    
    n: int = Field(
        1,
        description="Number of completions to generate",
        ge=1,
        le=5,
        example=1
    )
    
    stop: Optional[Union[str, List[str]]] = Field(
        None,
        description="Stop sequences where the API will stop generating",
        example=["\n", "Human:", "AI:"]
    )
    
    presence_penalty: float = Field(
        0.0,
        description="Positive values penalize new tokens based on their appearance in the text",
        ge=-2.0,
        le=2.0,
        example=0.1
    )
    
    frequency_penalty: float = Field(
        0.0,
        description="Positive values penalize new tokens based on their frequency in the text",
        ge=-2.0,
        le=2.0,
        example=0.1
    )
    
    logit_bias: Optional[Dict[str, float]] = Field(
        None,
        description="Modify the likelihood of specified tokens appearing in the completion",
        example={"42": -100, "43": -100}
    )
    
    user: Optional[str] = Field(
        None,
        description="User identifier for monitoring and rate limiting",
        example="user-123"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata to associate with the request",
        example={"session_id": "abc-123", "client": "mobile"}
    )
    
    @validator('messages')
    def validate_messages(cls, v):
        """Validate messages list."""
        if not v:
            raise ValueError("Messages cannot be empty")
        return v
    
    @validator('stop')
    def validate_stop(cls, v):
        """Validate stop sequences."""
        if v is not None:
            if isinstance(v, str):
                if len(v) > 100:
                    raise ValueError("Stop sequence too long (max 100 characters)")
                sequences = [v]
            elif isinstance(v, list):
                if len(v) > 4:
                    raise ValueError("Too many stop sequences (max 4)")
                sequences = v
                for seq in sequences:
                    if len(seq) > 100:
                        raise ValueError(f"Stop sequence too long: {seq[:20]}... (max 100 characters)")
            else:
                raise ValueError("stop must be string or list of strings")
        return v
    
    @validator('logit_bias')
    def validate_logit_bias(cls, v):
        """Validate logit bias values."""
        if v is not None:
            if len(v) > 100:
                raise ValueError("Too many logit bias entries (max 100)")
            for token_id, bias in v.items():
                try:
                    int(token_id)
                except ValueError:
                    raise ValueError(f"Token ID must be integer, got {token_id}")
                if bias < -100 or bias > 100:
                    raise ValueError(f"Logit bias must be between -100 and 100, got {bias}")
        return v
    
    @root_validator
    def validate_temperature_top_p(cls, values):
        """Validate temperature and top_p combination."""
        temp = values.get('temperature')
        top_p = values.get('top_p')
        
        # OpenAI recommends not changing both at the same time
        if temp != 1.0 and top_p != 1.0:
            import warnings
            warnings.warn("Changing both temperature and top_p is not recommended. Usually you should modify one at a time.")
        
        return values
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "model": "gpt-3.5-turbo",
                "stream": False,
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.95,
                "n": 1,
                "stop": ["\n"],
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "user": "user-123"
            }
        }


class ChatCompletionStreamRequest(ChatCompletionRequest):
    """
    Streaming chat completion request.
    Forces stream=True and adds streaming-specific parameters.
    """
    
    stream: bool = Field(
        True,
        description="Whether to stream the response (always true for streaming endpoint)",
        const=True
    )
    
    stream_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Options for streaming",
        example={"include_usage": True}
    )
    
    @validator('stream')
    def validate_stream(cls, v):
        """Ensure stream is True."""
        if not v:
            raise ValueError("Stream must be True for streaming endpoint")
        return v


# ============================================================================
# CHAT COMPLETION RESPONSE SCHEMAS
# ============================================================================

class ChatCompletionChoice(BaseModel):
    """
    Single completion choice in the response.
    """
    
    index: int = Field(
        ...,
        description="Choice index",
        example=0
    )
    message: ChatMessage = Field(
        ...,
        description="Generated message"
    )
    finish_reason: Optional[str] = Field(
        None,
        description="Reason why the completion stopped",
        example="stop"
    )
    
    @validator('finish_reason')
    def validate_finish_reason(cls, v):
        """Validate finish reason."""
        if v is not None:
            allowed_reasons = ['stop', 'length', 'content_filter', 'tool_calls', 'error']
            if v not in allowed_reasons:
                raise ValueError(f"Finish reason must be one of {allowed_reasons}")
        return v


class CompletionUsage(BaseModel):
    """
    Token usage statistics for the completion.
    """
    
    prompt_tokens: int = Field(
        ...,
        description="Number of tokens in the prompt",
        ge=0,
        example=10
    )
    completion_tokens: int = Field(
        ...,
        description="Number of tokens in the completion",
        ge=0,
        example=8
    )
    total_tokens: int = Field(
        ...,
        description="Total number of tokens used",
        ge=0,
        example=18
    )
    
    @root_validator
    def validate_total(cls, values):
        """Validate total tokens equals sum."""
        prompt = values.get('prompt_tokens', 0)
        completion = values.get('completion_tokens', 0)
        total = values.get('total_tokens', 0)
        
        if total != prompt + completion:
            values['total_tokens'] = prompt + completion
        
        return values


class ChatCompletionResponse(BaseModel):
    """
    Chat completion response schema.
    
    OpenAI-compatible format with ID, choices, and usage statistics.
    """
    
    id: str = Field(
        ...,
        description="Unique identifier for the completion",
        example="chatcmpl-123abc"
    )
    object: str = Field(
        "chat.completion",
        description="Object type",
        const=True
    )
    created: int = Field(
        ...,
        description="Unix timestamp of creation time",
        example=1705300000
    )
    model: str = Field(
        ...,
        description="Model used for completion",
        example="gpt-3.5-turbo"
    )
    choices: List[ChatCompletionChoice] = Field(
        ...,
        description="List of completion choices",
        min_items=1,
        max_items=5
    )
    usage: CompletionUsage = Field(
        ...,
        description="Token usage statistics"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the response",
        example={
            "cache_hit": False,
            "latency_ms": 450,
            "routing_strategy": "hybrid",
            "backend_type": "external"
        }
    )
    
    @validator('id')
    def validate_id(cls, v):
        """Validate completion ID format."""
        if not v.startswith('chatcmpl-'):
            raise ValueError("Completion ID must start with 'chatcmpl-'")
        return v
    
    @validator('created')
    def validate_created(cls, v):
        """Validate created timestamp."""
        import time
        current = int(time.time())
        if v > current + 300:  # 5 minutes in the future
            raise ValueError("Created timestamp cannot be in the future")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-123abc",
                "object": "chat.completion",
                "created": 1705300000,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The capital of France is Paris."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18
                },
                "metadata": {
                    "cache_hit": False,
                    "latency_ms": 450,
                    "routing_strategy": "hybrid"
                }
            }
        }


# ============================================================================
# STREAMING RESPONSE SCHEMAS
# ============================================================================

class ChatCompletionChunkChoice(BaseModel):
    """
    Single choice in a streaming chunk.
    """
    
    index: int = Field(
        ...,
        description="Choice index",
        example=0
    )
    delta: Dict[str, Any] = Field(
        ...,
        description="Delta update (partial message)",
        example={"role": "assistant", "content": "Paris"}
    )
    finish_reason: Optional[str] = Field(
        None,
        description="Reason why the completion stopped (final chunk only)",
        example=None
    )
    
    @validator('delta')
    def validate_delta(cls, v):
        """Validate delta contains valid fields."""
        if 'content' in v and not isinstance(v['content'], str):
            raise ValueError("Delta content must be string")
        if 'role' in v and v['role'] not in ['system', 'user', 'assistant']:
            raise ValueError("Delta role must be system, user, or assistant")
        return v


class ChatCompletionChunk(BaseModel):
    """
    Streaming chat completion chunk schema.
    
    OpenAI-compatible format for Server-Sent Events (SSE).
    """
    
    id: str = Field(
        ...,
        description="Unique identifier for the completion",
        example="chatcmpl-123abc"
    )
    object: str = Field(
        "chat.completion.chunk",
        description="Object type",
        const=True
    )
    created: int = Field(
        ...,
        description="Unix timestamp of creation time",
        example=1705300000
    )
    model: str = Field(
        ...,
        description="Model used for completion",
        example="gpt-3.5-turbo"
    )
    choices: List[ChatCompletionChunkChoice] = Field(
        ...,
        description="List of completion choices (usually 1)",
        min_items=1,
        max_items=1
    )
    
    @validator('id')
    def validate_id(cls, v):
        """Validate chunk ID format."""
        if not v.startswith('chatcmpl-'):
            raise ValueError("Chunk ID must start with 'chatcmpl-'")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-123abc",
                "object": "chat.completion.chunk",
                "created": 1705300000,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "Paris"
                        },
                        "finish_reason": None
                    }
                ]
            }
        }


# ============================================================================
# CHAT HISTORY SCHEMAS
# ============================================================================

class ChatHistoryEntry(BaseModel):
    """
    Chat history entry for database storage and retrieval.
    """
    
    id: str = Field(
        ...,
        description="Unique identifier for the chat entry"
    )
    conversation_id: str = Field(
        ...,
        description="Conversation identifier"
    )
    user_id: Optional[str] = Field(
        None,
        description="User identifier"
    )
    role: str = Field(
        ...,
        description="Message role"
    )
    content: str = Field(
        ...,
        description="Message content"
    )
    model: Optional[str] = Field(
        None,
        description="Model used for response"
    )
    tokens: int = Field(
        0,
        description="Token count"
    )
    latency_ms: float = Field(
        0,
        description="Response latency in milliseconds"
    )
    created_at: datetime = Field(
        ...,
        description="Creation timestamp"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chat-123abc",
                "conversation_id": "conv-456def",
                "user_id": "user-789",
                "role": "assistant",
                "content": "The capital of France is Paris.",
                "model": "gpt-3.5-turbo",
                "tokens": 8,
                "latency_ms": 450,
                "created_at": "2024-01-15T10:30:00Z",
                "metadata": {
                    "cache_hit": False,
                    "routing_strategy": "hybrid"
                }
            }
        }


class ChatHistoryResponse(BaseModel):
    """
    Chat history list response with pagination.
    """
    
    entries: List[ChatHistoryEntry] = Field(
        ...,
        description="List of chat history entries"
    )
    total: int = Field(
        ...,
        description="Total number of entries"
    )
    limit: int = Field(
        ...,
        description="Number of entries per page"
    )
    offset: int = Field(
        ...,
        description="Offset for pagination"
    )
    has_more: bool = Field(
        ...,
        description="Whether there are more entries"
    )


# ============================================================================
# CHAT FEEDBACK SCHEMAS
# ============================================================================

class ChatFeedbackRequest(BaseModel):
    """
    User feedback on a chat completion.
    """
    
    completion_id: str = Field(
        ...,
        description="ID of the completion to provide feedback for"
    )
    rating: int = Field(
        ...,
        description="Rating from 1-5",
        ge=1,
        le=5,
        example=5
    )
    appropriate_model: bool = Field(
        ...,
        description="Whether the selected model was appropriate",
        example=True
    )
    better_model: Optional[str] = Field(
        None,
        description="Model that would have been better",
        example="gpt-4"
    )
    comments: Optional[str] = Field(
        None,
        description="Additional comments",
        max_length=1000,
        example="The response was perfect and very fast!"
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Tags for categorizing feedback",
        example=["accuracy", "speed"]
    )
    
    @validator('completion_id')
    def validate_completion_id(cls, v):
        """Validate completion ID format."""
        if not v.startswith('chatcmpl-'):
            raise ValueError("Completion ID must start with 'chatcmpl-'")
        return v
    
    @validator('better_model')
    def validate_better_model(cls, v):
        """Validate better model suggestion."""
        if v is not None and len(v) > 100:
            raise ValueError("Model name too long")
        return v


class ChatFeedbackResponse(BaseModel):
    """
    Response to chat feedback submission.
    """
    
    status: str = Field(
        ...,
        description="Status of feedback submission",
        example="success"
    )
    message: str = Field(
        ...,
        description="Human-readable message",
        example="Feedback received. Thank you for helping improve our service!"
    )
    feedback_id: str = Field(
        ...,
        description="Unique identifier for the feedback"
    )
    timestamp: datetime = Field(
        ...,
        description="Submission timestamp"
    )


# ============================================================================
# CHAT ERROR SCHEMAS
# ============================================================================

class ChatErrorDetail(BaseModel):
    """
    Detailed error information for chat endpoints.
    """
    
    code: str = Field(
        ...,
        description="Error code",
        example="RATE_LIMIT_EXCEEDED"
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
        example="Rate limit exceeded. Please try again in 30 seconds."
    )
    param: Optional[str] = Field(
        None,
        description="Parameter that caused the error",
        example="max_tokens"
    )
    type: str = Field(
        ...,
        description="Error type",
        example="invalid_request_error"
    )


class ChatErrorResponse(BaseModel):
    """
    Error response for chat endpoints.
    """
    
    error: ChatErrorDetail = Field(
        ...,
        description="Error details"
    )
    request_id: str = Field(
        ...,
        description="Request ID for tracing",
        example="req-123abc"
    )
    timestamp: datetime = Field(
        ...,
        description="Error timestamp"
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Message schemas
    "ChatMessage",
    "ChatMessageList",
    
    # Request schemas
    "ChatCompletionRequest",
    "ChatCompletionStreamRequest",
    
    # Response schemas
    "ChatCompletionChoice",
    "CompletionUsage",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    
    # History schemas
    "ChatHistoryEntry",
    "ChatHistoryResponse",
    
    # Feedback schemas
    "ChatFeedbackRequest",
    "ChatFeedbackResponse",
    
    # Error schemas
    "ChatErrorDetail",
    "ChatErrorResponse"
]