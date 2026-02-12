"""
Chat Endpoints - Production Ready LLM Chat Interface
Handles chat completions with streaming, model routing, and caching
"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any, AsyncGenerator
import time
import uuid
import json
import asyncio
from pydantic import BaseModel, Field, validator

from core.exceptions import ModelNotAvailableError, InvalidPromptError, RateLimitExceededError
from core.logging import get_logger
from core.security import verify_api_key, get_current_user
from gateway.gateway_handler import GatewayHandler
from models.model_manager import ModelManager
from cache.cache_manager import CacheManager
from monitoring.metrics import MetricsCollector
from database.repositories.chat_repository import ChatRepository
from services.chat_service import ChatService
from schemas.chat import (
    ChatRequest, ChatResponse, ChatMessage, 
    ChatCompletionRequest, ChatCompletionResponse,
    StreamChunk
)

# Initialize router
router = APIRouter(prefix="/chat", tags=["Chat"])

# Initialize logger
logger = get_logger(__name__)

# Initialize services (will be injected in production)
gateway_handler = GatewayHandler()
chat_service = ChatService()
metrics_collector = MetricsCollector()
chat_repository = ChatRepository()

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class ChatRequest(BaseModel):
    """
    Chat completion request schema
    
    Why this structure?
    - OpenAI-compatible API for easy integration
    - Supports multiple messages for conversation context
    - Flexible model selection with auto-routing fallback
    - Streaming support for real-time responses
    - Temperature and token limits for response control
    """
    
    messages: List[ChatMessage] = Field(
        ...,
        description="List of messages in the conversation",
        min_items=1,
        example=[
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    
    model: Optional[str] = Field(
        None,
        description="Model ID to use. If not specified, auto-routing will be used",
        example="gpt-3.5-turbo"
    )
    
    stream: bool = Field(
        False,
        description="Whether to stream the response token by token"
    )
    
    max_tokens: Optional[int] = Field(
        1000,
        description="Maximum number of tokens to generate",
        ge=1,
        le=4096
    )
    
    temperature: float = Field(
        0.7,
        description="Sampling temperature (0-2). Higher values make output more random",
        ge=0.0,
        le=2.0
    )
    
    top_p: Optional[float] = Field(
        0.95,
        description="Nucleus sampling: only consider tokens with top_p probability",
        ge=0.0,
        le=1.0
    )
    
    n: Optional[int] = Field(
        1,
        description="Number of completions to generate",
        ge=1,
        le=5
    )
    
    stop: Optional[List[str]] = Field(
        None,
        description="Stop sequences where the API will stop generating",
        max_items=4
    )
    
    presence_penalty: Optional[float] = Field(
        0.0,
        description="Positive values penalize new tokens based on their appearance in the text",
        ge=-2.0,
        le=2.0
    )
    
    frequency_penalty: Optional[float] = Field(
        0.0,
        description="Positive values penalize new tokens based on their frequency in the text",
        ge=-2.0,
        le=2.0
    )
    
    user: Optional[str] = Field(
        None,
        description="User identifier for monitoring and rate limiting"
    )
    
    @validator('messages')
    def validate_messages(cls, v):
        """Validate message format and content"""
        if not v:
            raise ValueError('Messages cannot be empty')
        
        # Check for empty content
        for msg in v:
            if not msg.content or not msg.content.strip():
                raise ValueError('Message content cannot be empty')
            
            # Validate role
            if msg.role not in ['user', 'assistant', 'system']:
                raise ValueError(f'Invalid role: {msg.role}. Must be user, assistant, or system')
        
        return v


class ChatResponse(BaseModel):
    """
    Chat completion response schema
    
    Why this structure?
    - Matches OpenAI API format for compatibility
    - Includes usage statistics for cost tracking
    - Provides metadata for debugging and monitoring
    - Returns model information for transparency
    """
    
    id: str = Field(
        ...,
        description="Unique identifier for the completion",
        example="chatcmpl-123abc"
    )
    
    object: str = Field(
        "chat.completion",
        description="Object type"
    )
    
    created: int = Field(
        ...,
        description="Unix timestamp of creation time"
    )
    
    model: str = Field(
        ...,
        description="Model used for completion",
        example="gpt-3.5-turbo"
    )
    
    choices: List[Dict[str, Any]] = Field(
        ...,
        description="List of completion choices",
        example=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris."
            },
            "finish_reason": "stop"
        }]
    )
    
    usage: Dict[str, int] = Field(
        ...,
        description="Token usage statistics",
        example={
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the request",
        example={
            "cache_hit": False,
            "latency_ms": 450,
            "routing_strategy": "hybrid",
            "backend_type": "external"
        }
    )


class ChatMessage(BaseModel):
    """
    Chat message schema
    
    Why this structure?
    - Simple, clear role/content format
    - Supports system, user, and assistant roles
    - Optional name field for multi-user conversations
    """
    
    role: str = Field(
        ...,
        description="Role of the message sender",
        regex="^(system|user|assistant)$"
    )
    
    content: str = Field(
        ...,
        description="Content of the message",
        min_length=1
    )
    
    name: Optional[str] = Field(
        None,
        description="Name of the sender (for multi-user conversations)"
    )


class StreamChunk(BaseModel):
    """
    Streaming response chunk schema
    
    Why this structure?
    - Matches OpenAI streaming format
    - Each chunk contains one token or special event
    - Final chunk includes finish_reason
    """
    
    id: str = Field(..., description="Request ID")
    object: str = Field("chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="Choice data")


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/completions",
    response_model=ChatResponse,
    summary="Generate chat completion",
    description="""
    Generate a chat completion using the specified model or auto-routing.
    
    This endpoint provides:
    - **Intelligent model routing**: Automatically selects the best model based on your prompt
    - **Semantic caching**: Returns cached responses for similar prompts (85% similarity threshold)
    - **Streaming support**: Real-time token-by-token responses
    - **Multiple model support**: 15+ models including Grok, GPT-4, Claude, and local models
    - **Cost optimization**: Routes to cost-effective models when quality requirements are met
    
    ### Model Selection
    - If `model` is specified: Uses that exact model
    - If `model` is omitted: Auto-routing selects optimal model based on:
      - Prompt complexity and type
      - Latency requirements
      - Cost constraints
      - Model availability and health
    
    ### Caching
    Responses are cached with semantic similarity matching:
    - Exact matches: Immediate cache hit
    - Similar prompts (85%+ similarity): Cached response with confidence score
    - Cache TTL: 1 hour (configurable)
    
    ### Rate Limits
    - Free tier: 100 requests/minute
    - Premium: Custom limits based on API key
    
    ### Error Handling
    - Automatic fallback to backup models on failure
    - Circuit breaker prevents cascading failures
    - Detailed error messages with request IDs for debugging
    """,
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Invalid request (empty prompt, invalid parameters)"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Model not available or service busy"}
    }
)
async def chat_completion(
    request: ChatRequest,
    req: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate chat completion with intelligent routing
    
    Why this endpoint structure?
    1. **Separation of concerns**: Request validation, processing, and response handled separately
    2. **Background tasks**: Non-critical operations (logging, metrics) run asynchronously
    3. **Error handling**: Comprehensive error handling with specific status codes
    4. **Performance tracking**: Latency monitoring built in
    5. **User identification**: API key authentication for rate limiting and tracking
    """
    
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        "chat_request_received",
        request_id=request_id,
        model=request.model,
        stream=request.stream,
        message_count=len(request.messages),
        user=request.user
    )
    
    try:
        # ===== STEP 1: Extract user prompt =====
        # Get the last user message or combine all messages
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise InvalidPromptError("No user message found in conversation")
        
        prompt = user_messages[-1].content
        
        # ===== STEP 2: Check cache (if enabled) =====
        cache_hit = False
        cached_response = None
        
        if settings.cache_enabled:
            cache_key = await chat_service.generate_cache_key(request)
            cached_response = await chat_service.check_cache(cache_key, prompt)
            
            if cached_response:
                cache_hit = True
                logger.info(
                    "cache_hit",
                    request_id=request_id,
                    cache_key=cache_key,
                    similarity=cached_response.get("similarity", 1.0)
                )
                
                # Record cache hit metrics
                background_tasks.add_task(
                    metrics_collector.record_cache_hit,
                    model=cached_response.get("model", "unknown")
                )
                
                # Return cached response
                response = ChatResponse(
                    id=f"chatcmpl-{request_id[:8]}",
                    created=int(time.time()),
                    model=cached_response["model"],
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": cached_response["content"]
                        },
                        "finish_reason": "stop"
                    }],
                    usage=cached_response["usage"],
                    metadata={
                        "cache_hit": True,
                        "similarity": cached_response.get("similarity", 1.0),
                        "latency_ms": round((time.time() - start_time) * 1000, 2),
                        "request_id": request_id
                    }
                )
                
                # Log request asynchronously
                background_tasks.add_task(
                    chat_repository.log_request,
                    request_id=request_id,
                    user_id=req.state.user_id if hasattr(req.state, 'user_id') else None,
                    prompt=prompt,
                    response=cached_response["content"],
                    model=cached_response["model"],
                    tokens=cached_response["usage"]["total_tokens"],
                    latency_ms=(time.time() - start_time) * 1000,
                    cache_hit=True
                )
                
                return response
        
        # ===== STEP 3: Process request through gateway =====
        # This handles model selection, routing, and inference
        result = await gateway_handler.process_request(
            request_id=request_id,
            prompt=prompt,
            messages=request.messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            user=request.user
        )
        
        # ===== STEP 4: Cache the response (if enabled) =====
        if settings.cache_enabled and not cache_hit:
            cache_key = await chat_service.generate_cache_key(request)
            background_tasks.add_task(
                chat_service.cache_response,
                cache_key=cache_key,
                prompt=prompt,
                response=result["content"],
                model=result["model"],
                usage=result["usage"]
            )
        
        # ===== STEP 5: Build response =====
        response = ChatResponse(
            id=f"chatcmpl-{request_id[:8]}",
            created=int(time.time()),
            model=result["model"],
            choices=[{
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": result["content"] if i == 0 else result.get(f"content_{i}", "")
                },
                "finish_reason": result.get("finish_reason", "stop")
            } for i in range(result.get("n", 1))],
            usage=result["usage"],
            metadata={
                "cache_hit": cache_hit,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "request_id": request_id,
                "routing_strategy": result.get("routing_strategy", "auto"),
                "backend_type": result.get("backend_type", "unknown"),
                "fallback_used": result.get("fallback_used", False)
            }
        )
        
        # ===== STEP 6: Log request asynchronously =====
        background_tasks.add_task(
            chat_repository.log_request,
            request_id=request_id,
            user_id=req.state.user_id if hasattr(req.state, 'user_id') else None,
            prompt=prompt,
            response=result["content"],
            model=result["model"],
            tokens=result["usage"]["total_tokens"],
            latency_ms=(time.time() - start_time) * 1000,
            cache_hit=cache_hit,
            metadata={
                "routing_strategy": result.get("routing_strategy"),
                "backend_type": result.get("backend_type"),
                "fallback_used": result.get("fallback_used", False)
            }
        )
        
        # ===== STEP 7: Record metrics =====
        background_tasks.add_task(
            metrics_collector.record_chat_completion,
            model=result["model"],
            tokens=result["usage"]["total_tokens"],
            latency_ms=(time.time() - start_time) * 1000,
            success=True
        )
        
        logger.info(
            "chat_request_completed",
            request_id=request_id,
            model=result["model"],
            tokens=result["usage"]["total_tokens"],
            latency_ms=round((time.time() - start_time) * 1000, 2),
            cache_hit=cache_hit
        )
        
        return response
        
    except ModelNotAvailableError as e:
        logger.error(
            "model_not_available",
            request_id=request_id,
            model=request.model,
            error=str(e)
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ModelNotAvailable",
                "message": str(e),
                "request_id": request_id,
                "timestamp": time.time()
            }
        )
        
    except RateLimitExceededError as e:
        logger.warning(
            "rate_limit_exceeded",
            request_id=request_id,
            user=request.user
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "RateLimitExceeded",
                "message": "Rate limit exceeded. Please try again later.",
                "request_id": request_id,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(
            "chat_request_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        # Record error metrics
        background_tasks.add_task(
            metrics_collector.record_chat_completion,
            model=request.model or "unknown",
            tokens=0,
            latency_ms=(time.time() - start_time) * 1000,
            success=False,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "An error occurred while processing your request",
                "request_id": request_id,
                "timestamp": time.time()
            }
        )


@router.post(
    "/completions/stream",
    summary="Stream chat completion",
    description="""
    Stream chat completions token by token in real-time.
    
    Benefits of streaming:
    - **Instant feedback**: Users see responses as they're generated
    - **Better UX**: Typing effect improves perceived performance
    - **Early cancellation**: Can stop generation mid-way
    - **Lower latency**: First token arrives faster
    
    The stream returns Server-Sent Events (SSE) with each chunk containing:
    1. A single token or special token
    2. The final chunk with finish_reason
    
    Use the `stream` parameter with `curl`:
    ```bash
    curl -X POST http://localhost:8000/api/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -H "X-API-Key: your-api-key" \\
      -d '{
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true
      }'
    ```
    """,
    responses={
        200: {
            "description": "Streaming response",
            "content": {
                "text/event-stream": {
                    "example": "data: {\"id\":\"chatcmpl-123\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
                }
            }
        }
    }
)
async def chat_completion_stream(
    request: ChatRequest,
    req: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Stream chat completions with Server-Sent Events
    
    Why streaming implementation?
    1. **Async generator**: Yields tokens as they're generated
    2. **SSE format**: Standard streaming format with 'data:' prefix
    3. **Error handling**: Can send errors within stream
    4. **Resource efficient**: No need to store full response in memory
    """
    
    if not request.stream:
        raise HTTPException(
            status_code=400,
            detail="Stream must be set to true for streaming endpoint"
        )
    
    request_id = str(uuid.uuid4())
    
    logger.info(
        "stream_request_started",
        request_id=request_id,
        model=request.model
    )
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        
        try:
            # Get the last user message
            user_messages = [msg for msg in request.messages if msg.role == "user"]
            if not user_messages:
                error_chunk = StreamChunk(
                    id=f"chatcmpl-{request_id[:8]}",
                    created=int(time.time()),
                    model=request.model or "unknown",
                    choices=[{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "error"
                    }]
                )
                yield f"data: {json.dumps(error_chunk.dict())}\n\n"
                yield f"data: [DONE]\n\n"
                return
            
            prompt = user_messages[-1].content
            
            # Process streaming through gateway
            async for token in gateway_handler.process_streaming_request(
                request_id=request_id,
                prompt=prompt,
                messages=request.messages,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                # Create chunk with token
                chunk = StreamChunk(
                    id=f"chatcmpl-{request_id[:8]}",
                    created=int(time.time()),
                    model=token.get("model", request.model or "unknown"),
                    choices=[{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": token.get("content", "")
                        },
                        "finish_reason": token.get("finish_reason")
                    }]
                )
                
                yield f"data: {json.dumps(chunk.dict())}\n\n"
                
                # Small delay to simulate natural typing
                await asyncio.sleep(0.01)
            
            # Send completion signal
            yield f"data: [DONE]\n\n"
            
            # Log request asynchronously
            background_tasks.add_task(
                chat_repository.log_request,
                request_id=request_id,
                user_id=req.state.user_id if hasattr(req.state, 'user_id') else None,
                prompt=prompt,
                response="",  # Stream doesn't store full response
                model=request.model or "auto",
                tokens=0,  # TODO: Track tokens in streaming
                latency_ms=0,
                cache_hit=False,
                metadata={"streaming": True}
            )
            
            logger.info(
                "stream_request_completed",
                request_id=request_id,
                model=request.model
            )
            
        except Exception as e:
            logger.error(
                "stream_request_failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            
            # Send error chunk
            error_chunk = StreamChunk(
                id=f"chatcmpl-{request_id[:8]}",
                created=int(time.time()),
                model=request.model or "unknown",
                choices=[{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error"
                }]
            )
            yield f"data: {json.dumps(error_chunk.dict())}\n\n"
            yield f"data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-Request-ID": request_id
        }
    )


@router.get(
    "/history",
    summary="Get chat history",
    description="Retrieve paginated chat history for the authenticated user"
)
async def get_chat_history(
    req: Request,
    limit: int = 50,
    offset: int = 0,
    conversation_id: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """Get chat history with pagination"""
    
    user_id = getattr(req.state, 'user_id', None)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    
    history = await chat_repository.get_user_history(
        user_id=user_id,
        limit=limit,
        offset=offset,
        conversation_id=conversation_id
    )
    
    return {
        "history": history,
        "total": len(history),
        "limit": limit,
        "offset": offset
    }


@router.delete(
    "/history/{conversation_id}",
    summary="Delete conversation",
    description="Delete a specific conversation from history"
)
async def delete_conversation(
    conversation_id: str,
    req: Request,
    api_key: str = Depends(verify_api_key)
):
    """Delete a conversation"""
    
    user_id = getattr(req.state, 'user_id', None)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    
    success = await chat_repository.delete_conversation(
        conversation_id=conversation_id,
        user_id=user_id
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )
    
    return {
        "success": True,
        "message": f"Conversation {conversation_id} deleted"
    }


@router.get(
    "/models",
    summary="Get available chat models",
    description="List all available models for chat completions"
)
async def get_chat_models(
    api_key: str = Depends(verify_api_key)
):
    """Get list of available chat models"""
    
    models = await ModelManager.get_available_models(
        capabilities=["chat", "instruction"]
    )
    
    return {
        "object": "list",
        "data": models
    }


@router.post(
    "/feedback",
    summary="Submit feedback",
    description="Submit user feedback for a chat completion"
)
async def submit_feedback(
    request: Request,
    completion_id: str,
    feedback: Dict[str, Any],
    api_key: str = Depends(verify_api_key)
):
    """Submit user feedback for model improvement"""
    
    # TODO: Store feedback in database
    # TODO: Use feedback for routing optimization
    # TODO: Implement RLHF pipeline
    
    logger.info(
        "feedback_received",
        completion_id=completion_id,
        feedback=feedback
    )
    
    return {
        "success": True,
        "message": "Feedback received. Thank you!"
    }