"""
Response Handler - Production Ready
Comprehensive response processing for LLM inference with streaming support,
format conversion, error handling, and response optimization.
"""

import json
import time
import asyncio
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from datetime import datetime
import gzip
import io

from fastapi import Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from core.logging import get_logger
from core.exceptions import LLMGatewayException
from config import settings

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str
    message: str
    request_id: str
    timestamp: str
    status_code: int


class ChatCompletionChoice(BaseModel):
    """Chat completion choice schema."""
    index: int
    message: Dict[str, str]
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response schema."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


class ChatCompletionChunk(BaseModel):
    """Streaming chat completion chunk schema."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


class EmbeddingResponse(BaseModel):
    """Embedding response schema."""
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]


# ============================================================================
# RESPONSE FORMATTER
# ============================================================================

class ResponseFormatter:
    """
    Format LLM responses into standard API responses.
    
    Features:
    - OpenAI-compatible format
    - Multiple response types (chat, embedding, completion)
    - Consistent error structure
    - Metadata injection
    """
    
    @staticmethod
    def format_chat_response(
        request_id: str,
        model: str,
        content: Union[str, List[str]],
        tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        finish_reason: str = "stop",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format chat completion response (OpenAI compatible).
        
        Args:
            request_id: Unique request identifier
            model: Model ID used
            content: Generated response(s)
            tokens: Total tokens used
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion
            finish_reason: Reason for stopping
            metadata: Additional metadata
        
        Returns:
            Formatted chat completion response
        """
        created = int(time.time())
        response_id = f"chatcmpl-{request_id[:8]}"
        
        # Handle multiple completions
        if isinstance(content, list):
            choices = []
            for i, text in enumerate(content):
                choices.append({
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": finish_reason
                })
        else:
            choices = [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": finish_reason
            }]
        
        response = {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens or tokens // 2,
                "completion_tokens": completion_tokens or tokens // 2,
                "total_tokens": tokens
            }
        }
        
        # Add metadata if provided
        if metadata:
            response["metadata"] = metadata
        
        return response
    
    @staticmethod
    def format_streaming_chunk(
        request_id: str,
        model: str,
        content: str,
        finish_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format streaming response chunk (OpenAI compatible).
        
        Args:
            request_id: Unique request identifier
            model: Model ID used
            content: Response chunk
            finish_reason: Reason for stopping (final chunk only)
        
        Returns:
            Formatted streaming chunk
        """
        created = int(time.time())
        response_id = f"chatcmpl-{request_id[:8]}"
        
        delta = {}
        if content:
            delta["role"] = "assistant"
            delta["content"] = content
        
        return {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
    
    @staticmethod
    def format_embedding_response(
        request_id: str,
        model: str,
        embeddings: List[List[float]],
        tokens: int = 0
    ) -> Dict[str, Any]:
        """
        Format embedding response (OpenAI compatible).
        
        Args:
            request_id: Unique request identifier
            model: Model ID used
            embeddings: Generated embeddings
            tokens: Total tokens used
        
        Returns:
            Formatted embedding response
        """
        response_id = f"emb-{request_id[:8]}"
        
        data = []
        for i, embedding in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding
            })
        
        return {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {
                "prompt_tokens": tokens,
                "total_tokens": tokens
            }
        }
    
    @staticmethod
    def format_error(
        request_id: str,
        error: Union[str, Exception],
        status_code: int = 500,
        error_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format error response.
        
        Args:
            request_id: Unique request identifier
            error: Error message or exception
            status_code: HTTP status code
            error_type: Error type/code
        
        Returns:
            Formatted error response
        """
        if isinstance(error, Exception):
            error_message = str(error)
            error_type = error_type or error.__class__.__name__
        else:
            error_message = error
            error_type = error_type or "InternalServerError"
        
        return {
            "error": {
                "code": error_type,
                "message": error_message,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status_code": status_code
            }
        }


# ============================================================================
# RESPONSE OPTIMIZER
# ============================================================================

class ResponseOptimizer:
    """
    Optimize responses for size and speed.
    
    Features:
    - Compression (gzip, brotli)
    - Response caching
    - Field filtering
    - Truncation
    """
    
    def __init__(self):
        self.min_compress_size = 1024  # 1KB
        self.default_encoding = "utf-8"
    
    async def optimize(
        self,
        response: Dict[str, Any],
        accept_encoding: Optional[str] = None,
        fields: Optional[List[str]] = None,
        max_size: Optional[int] = None
    ) -> Union[Dict[str, Any], Response]:
        """
        Optimize response for delivery.
        
        Args:
            response: Response data
            accept_encoding: Accepted compression algorithms
            fields: Fields to include (field filtering)
            max_size: Maximum response size in bytes
        
        Returns:
            Optimized response
        """
        # Field filtering
        if fields:
            response = self._filter_fields(response, fields)
        
        # Truncation
        if max_size:
            response = self._truncate_response(response, max_size)
        
        return response
    
    def _filter_fields(
        self,
        data: Dict[str, Any],
        fields: List[str]
    ) -> Dict[str, Any]:
        """Filter response to include only specified fields."""
        filtered = {}
        
        for field in fields:
            if field in data:
                filtered[field] = data[field]
        
        return filtered
    
    def _truncate_response(
        self,
        data: Dict[str, Any],
        max_size: int
    ) -> Dict[str, Any]:
        """Truncate response to fit within size limit."""
        # Convert to JSON to check size
        json_str = json.dumps(data)
        current_size = len(json_str.encode(self.default_encoding))
        
        if current_size <= max_size:
            return data
        
        # Truncate long text fields
        if "choices" in data:
            for choice in data["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    # Calculate how much we need to truncate
                    ratio = max_size / current_size
                    new_length = int(len(content) * ratio * 0.8)  # 80% of ratio
                    choice["message"]["content"] = content[:new_length] + "..."
        
        return data
    
    def compress(
        self,
        data: Union[Dict[str, Any], str, bytes],
        encoding: str = "gzip"
    ) -> bytes:
        """
        Compress response data.
        
        Args:
            data: Response data
            encoding: Compression algorithm (gzip, br)
        
        Returns:
            Compressed bytes
        """
        if isinstance(data, dict):
            data = json.dumps(data)
        
        if isinstance(data, str):
            data = data.encode(self.default_encoding)
        
        if encoding == "gzip":
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
                f.write(data)
            return buffer.getvalue()
        elif encoding == "br":
            import brotli
            return brotli.compress(data)
        else:
            return data


# ============================================================================
# STREAMING HANDLER
# ============================================================================

class StreamingHandler:
    """
    Handle streaming responses with Server-Sent Events (SSE).
    
    Features:
    - Token-by-token streaming
    - Heartbeat for keep-alive
    - Error injection
    - Flow control
    """
    
    def __init__(self):
        self.heartbeat_interval = 15  # seconds
        self.chunk_size = 1024  # bytes
    
    async def stream_chat_completion(
        self,
        request_id: str,
        model: str,
        token_generator: AsyncGenerator[str, None],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion tokens as SSE.
        
        Args:
            request_id: Unique request identifier
            model: Model ID
            token_generator: Async generator yielding tokens
            metadata: Additional metadata
        
        Yields:
            SSE formatted messages
        """
        formatter = ResponseFormatter()
        heartbeat_task = None
        
        try:
            # Start heartbeat for keep-alive
            heartbeat_task = asyncio.create_task(
                self._send_heartbeat()
            )
            
            # Stream tokens
            async for token in token_generator:
                chunk = formatter.format_streaming_chunk(
                    request_id=request_id,
                    model=model,
                    content=token
                )
                
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send final chunk with finish reason
            final_chunk = formatter.format_streaming_chunk(
                request_id=request_id,
                model=model,
                content="",
                finish_reason="stop"
            )
            
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except asyncio.CancelledError:
            logger.info(
                "streaming_cancelled",
                request_id=request_id,
                model=model
            )
            raise
            
        except Exception as e:
            # Send error as SSE
            error_chunk = formatter.format_streaming_chunk(
                request_id=request_id,
                model=model,
                content="",
                finish_reason="error"
            )
            
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield f"event: error\ndata: {str(e)}\n\n"
            
            raise
            
        finally:
            if heartbeat_task:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
    
    async def _send_heartbeat(self) -> None:
        """Send heartbeat comments to keep connection alive."""
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            yield ": heartbeat\n\n"


# ============================================================================
# RESPONSE HANDLER
# ============================================================================

class ResponseHandler:
    """
    Main response handler for the gateway.
    
    Handles:
    - Response formatting
    - Streaming responses
    - Error responses
    - Response optimization
    - Headers management
    """
    
    def __init__(self):
        self.formatter = ResponseFormatter()
        self.optimizer = ResponseOptimizer()
        self.streaming_handler = StreamingHandler()
        
        # Default headers
        self.default_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        logger.info("response_handler_initialized")
    
    # ========================================================================
    # SUCCESS RESPONSES
    # ========================================================================
    
    def create_chat_response(
        self,
        request_id: str,
        model: str,
        content: Union[str, List[str]],
        tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        finish_reason: str = "stop",
        metadata: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> JSONResponse:
        """
        Create a chat completion response.
        
        Args:
            request_id: Unique request identifier
            model: Model ID used
            content: Generated response(s)
            tokens: Total tokens used
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion
            finish_reason: Reason for stopping
            metadata: Additional metadata
            headers: Additional HTTP headers
        
        Returns:
            JSON response
        """
        response_data = self.formatter.format_chat_response(
            request_id=request_id,
            model=model,
            content=content,
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            metadata=metadata
        )
        
        return self._create_json_response(
            data=response_data,
            status_code=200,
            headers=headers
        )
    
    def create_streaming_response(
        self,
        request_id: str,
        model: str,
        token_generator: AsyncGenerator[str, None],
        metadata: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> StreamingResponse:
        """
        Create a streaming chat completion response.
        
        Args:
            request_id: Unique request identifier
            model: Model ID used
            token_generator: Async generator yielding tokens
            metadata: Additional metadata
            headers: Additional HTTP headers
        
        Returns:
            Streaming response
        """
        stream_generator = self.streaming_handler.stream_chat_completion(
            request_id=request_id,
            model=model,
            token_generator=token_generator,
            metadata=metadata
        )
        
        response_headers = self._merge_headers({
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            **self.default_headers
        }, headers)
        
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers=response_headers
        )
    
    def create_embedding_response(
        self,
        request_id: str,
        model: str,
        embeddings: List[List[float]],
        tokens: int = 0,
        headers: Optional[Dict[str, str]] = None
    ) -> JSONResponse:
        """
        Create an embedding response.
        
        Args:
            request_id: Unique request identifier
            model: Model ID used
            embeddings: Generated embeddings
            tokens: Total tokens used
            headers: Additional HTTP headers
        
        Returns:
            JSON response
        """
        response_data = self.formatter.format_embedding_response(
            request_id=request_id,
            model=model,
            embeddings=embeddings,
            tokens=tokens
        )
        
        return self._create_json_response(
            data=response_data,
            status_code=200,
            headers=headers
        )
    
    # ========================================================================
    # ERROR RESPONSES
    # ========================================================================
    
    def create_error_response(
        self,
        request_id: str,
        error: Union[str, Exception],
        status_code: int = 500,
        error_type: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> JSONResponse:
        """
        Create an error response.
        
        Args:
            request_id: Unique request identifier
            error: Error message or exception
            status_code: HTTP status code
            error_type: Error type/code
            headers: Additional HTTP headers
        
        Returns:
            JSON error response
        """
        response_data = self.formatter.format_error(
            request_id=request_id,
            error=error,
            status_code=status_code,
            error_type=error_type
        )
        
        # Log error
        logger.error(
            "error_response_created",
            request_id=request_id,
            status_code=status_code,
            error_type=error_type or error.__class__.__name__ if isinstance(error, Exception) else "Error"
        )
        
        return self._create_json_response(
            data=response_data,
            status_code=status_code,
            headers=headers
        )
    
    def create_validation_error_response(
        self,
        request_id: str,
        errors: List[Dict[str, Any]],
        headers: Optional[Dict[str, str]] = None
    ) -> JSONResponse:
        """
        Create a validation error response.
        
        Args:
            request_id: Unique request identifier
            errors: Validation error details
            headers: Additional HTTP headers
        
        Returns:
            JSON error response
        """
        response_data = {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": errors,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status_code": 422
            }
        }
        
        logger.warning(
            "validation_error_response",
            request_id=request_id,
            error_count=len(errors)
        )
        
        return self._create_json_response(
            data=response_data,
            status_code=422,
            headers=headers
        )
    
    def create_rate_limit_response(
        self,
        request_id: str,
        limit: int,
        remaining: int,
        reset_time: int,
        headers: Optional[Dict[str, str]] = None
    ) -> JSONResponse:
        """
        Create a rate limit exceeded response.
        
        Args:
            request_id: Unique request identifier
            limit: Rate limit
            remaining: Remaining requests
            reset_time: When limit resets (timestamp)
            headers: Additional HTTP headers
        
        Returns:
            JSON error response
        """
        response_data = {
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Rate limit exceeded. Please try again later.",
                "limit": limit,
                "remaining": remaining,
                "reset": reset_time,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status_code": 429
            }
        }
        
        rate_limit_headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
            "Retry-After": str(max(0, reset_time - int(time.time())))
        }
        
        return self._create_json_response(
            data=response_data,
            status_code=429,
            headers=self._merge_headers(rate_limit_headers, headers)
        )
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _create_json_response(
        self,
        data: Dict[str, Any],
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None
    ) -> JSONResponse:
        """
        Create a JSON response with standard headers.
        
        Args:
            data: Response data
            status_code: HTTP status code
            headers: Additional HTTP headers
        
        Returns:
            JSON response
        """
        response_headers = self._merge_headers(self.default_headers, headers)
        
        return JSONResponse(
            content=data,
            status_code=status_code,
            headers=response_headers
        )
    
    def _merge_headers(
        self,
        base_headers: Dict[str, str],
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Merge additional headers into base headers."""
        if not additional_headers:
            return base_headers.copy()
        
        merged = base_headers.copy()
        merged.update(additional_headers)
        return merged
    
    def add_cors_headers(
        self,
        headers: Dict[str, str],
        origin: Optional[str] = None
    ) -> Dict[str, str]:
        """Add CORS headers to response."""
        headers["Access-Control-Allow-Origin"] = origin or "*"
        headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
        headers["Access-Control-Expose-Headers"] = "X-Request-ID, X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset"
        headers["Access-Control-Max-Age"] = "600"
        
        return headers
    
    def add_security_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Add security headers to response."""
        headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        })
        
        # Add HSTS in production
        if settings.environment.is_production():
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        return headers


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_response_handler = None


def get_response_handler() -> ResponseHandler:
    """Get singleton response handler instance."""
    global _response_handler
    if not _response_handler:
        _response_handler = ResponseHandler()
    return _response_handler


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ResponseHandler",
    "ResponseFormatter",
    "ResponseOptimizer",
    "StreamingHandler",
    "get_response_handler",
    
    # Response schemas
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "EmbeddingResponse",
    "ErrorResponse"
]