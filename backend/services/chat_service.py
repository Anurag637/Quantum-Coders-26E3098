"""
Chat Service - Production Ready
Orchestrates chat completion workflows with streaming support,
context management, history tracking, and feedback integration.
"""

import asyncio
import uuid
import hashlib
from typing import Dict, Any, Optional, List, AsyncGenerator, Tuple
from datetime import datetime
import json

from core.logging import get_logger, get_request_logger
from core.exceptions import (
    ModelNotAvailableError,
    InvalidPromptError,
    ValidationError,
    RateLimitExceededError
)
from config import settings
from models.model_manager import ModelManager
from models.model_registry import ModelRegistry
from gateway.gateway_handler import GatewayHandler
from gateway.request_validator import RequestValidator
from gateway.response_handler import ResponseHandler
from router.prompt_router import PromptRouter
from router.prompt_analyzer import PromptAnalyzer
from cache.cache_manager import CacheManager
from cache.semantic_cache import SemanticCache
from monitoring.metrics import MetricsCollector
from database.repositories.chat_repository import ChatRepository
from database.repositories.request_log_repository import RequestLogRepository
from database.repositories.routing_repository import RoutingRepository
from services.routing_service import RoutingService
from services.token_counter import TokenCounter
from services.cost_calculator import CostCalculator

# Initialize logger
logger = get_logger(__name__)
request_logger = get_request_logger()

# ============================================================================
# CHAT SERVICE
# ============================================================================

class ChatService:
    """
    Chat service for LLM interactions.
    
    Features:
    - Chat completion with streaming
    - Conversation context management
    - Chat history persistence
    - User feedback collection
    - Conversation summarization
    - Multi-turn conversation support
    - Cost tracking per conversation
    """
    
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        model_registry: Optional[ModelRegistry] = None,
        gateway_handler: Optional[GatewayHandler] = None,
        request_validator: Optional[RequestValidator] = None,
        response_handler: Optional[ResponseHandler] = None,
        prompt_router: Optional[PromptRouter] = None,
        prompt_analyzer: Optional[PromptAnalyzer] = None,
        cache_manager: Optional[CacheManager] = None,
        semantic_cache: Optional[SemanticCache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        chat_repository: Optional[ChatRepository] = None,
        request_log_repository: Optional[RequestLogRepository] = None,
        routing_repository: Optional[RoutingRepository] = None,
        routing_service: Optional[RoutingService] = None
    ):
        # Initialize dependencies
        self.model_manager = model_manager or ModelManager()
        self.model_registry = model_registry or ModelRegistry()
        self.gateway_handler = gateway_handler or GatewayHandler()
        self.request_validator = request_validator or RequestValidator()
        self.response_handler = response_handler or ResponseHandler()
        self.prompt_router = prompt_router or PromptRouter()
        self.prompt_analyzer = prompt_analyzer or PromptAnalyzer()
        self.cache_manager = cache_manager or CacheManager()
        self.semantic_cache = semantic_cache or SemanticCache()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.chat_repository = chat_repository or ChatRepository()
        self.request_log_repository = request_log_repository or RequestLogRepository()
        self.routing_repository = routing_repository or RoutingRepository()
        self.routing_service = routing_service or RoutingService()
        
        # Initialize utilities
        self.token_counter = TokenCounter()
        self.cost_calculator = CostCalculator()
        
        # Active conversations cache
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        
        # Conversation context window
        self.max_context_messages = 50
        self.max_context_tokens = 16000
        
        # Statistics
        self.stats = {
            "total_chats": 0,
            "total_messages": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_latency_ms": 0.0,
            "avg_tokens_per_message": 0
        }
        
        logger.info("chat_service_initialized")
    
    # ========================================================================
    # CHAT COMPLETION
    # ========================================================================
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        n: int = 1,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        user: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate chat completion.
        
        Args:
            messages: List of chat messages
            model: Model to use (optional, auto-routed if not specified)
            stream: Whether to stream the response
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stop: Stop sequences
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            user: User identifier
            conversation_id: Existing conversation ID
            metadata: Additional metadata
        
        Returns:
            Chat completion response
        """
        start_time = datetime.utcnow()
        request_id = str(uuid.uuid4())
        
        self.stats["total_chats"] += 1
        self.stats["total_messages"] += len(messages)
        
        logger.info(
            "chat_completion_started",
            request_id=request_id,
            message_count=len(messages),
            model=model,
            stream=stream,
            conversation_id=conversation_id,
            user=user
        )
        
        try:
            # ====================================================================
            # STEP 1: Validate request
            # ====================================================================
            validated_request = await self.request_validator.validate_chat_request(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
            
            # ====================================================================
            # STEP 2: Get or create conversation
            # ====================================================================
            if not conversation_id:
                conversation_id = self._generate_conversation_id(user)
            
            # Load conversation context
            conversation = await self._get_conversation_context(
                conversation_id=conversation_id,
                user=user
            )
            
            # ====================================================================
            # STEP 3: Prepare full context
            # ====================================================================
            full_messages = await self._prepare_conversation_context(
                conversation=conversation,
                new_messages=messages
            )
            
            # Extract the latest user prompt
            prompt = self._extract_prompt(messages)
            
            # ====================================================================
            # STEP 4: Process through gateway
            # ====================================================================
            if stream:
                response = await self.gateway_handler.process_streaming_request(
                    request_id=request_id,
                    prompt=prompt,
                    messages=full_messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    user=user,
                    metadata=metadata
                )
                
                # For streaming, we need to handle the async generator
                # This is handled separately in the streaming endpoint
                return response
            else:
                response = await self.gateway_handler.process_request(
                    request_id=request_id,
                    prompt=prompt,
                    messages=full_messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    user=user,
                    metadata=metadata
                )
            
            # ====================================================================
            # STEP 5: Save to conversation history
            # ====================================================================
            await self._save_to_history(
                conversation_id=conversation_id,
                user_id=user,
                messages=messages,
                response=response,
                model=response["model"],
                tokens=response["tokens"],
                latency_ms=response["latency_ms"],
                cache_hit=response.get("cache_hit", False),
                metadata={
                    "request_id": request_id,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **metadata if metadata else {}
                }
            )
            
            # ====================================================================
            # STEP 6: Update conversation context
            # ====================================================================
            await self._update_conversation_context(
                conversation_id=conversation_id,
                user=user,
                messages=messages,
                response=response
            )
            
            # ====================================================================
            # STEP 7: Update statistics
            # ====================================================================
            await self._update_statistics(response)
            
            # ====================================================================
            # STEP 8: Record metrics
            # ====================================================================
            await self._record_metrics(response)
            
            # Add conversation ID to response
            response["conversation_id"] = conversation_id
            
            logger.info(
                "chat_completion_completed",
                request_id=request_id,
                conversation_id=conversation_id,
                model=response["model"],
                tokens=response["tokens"],
                latency_ms=response["latency_ms"],
                cache_hit=response.get("cache_hit", False)
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "chat_completion_failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        user: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion tokens.
        
        Args:
            messages: List of chat messages
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            user: User identifier
            conversation_id: Existing conversation ID
            metadata: Additional metadata
        
        Yields:
            Server-Sent Events formatted chunks
        """
        request_id = str(uuid.uuid4())
        
        logger.info(
            "chat_stream_started",
            request_id=request_id,
            message_count=len(messages),
            model=model,
            conversation_id=conversation_id,
            user=user
        )
        
        try:
            # Get or create conversation
            if not conversation_id:
                conversation_id = self._generate_conversation_id(user)
            
            # Load conversation context
            conversation = await self._get_conversation_context(
                conversation_id=conversation_id,
                user=user
            )
            
            # Prepare full context
            full_messages = await self._prepare_conversation_context(
                conversation=conversation,
                new_messages=messages
            )
            
            # Extract prompt
            prompt = self._extract_prompt(messages)
            
            # Get streaming response from gateway
            token_generator = self.gateway_handler.process_streaming_request(
                request_id=request_id,
                prompt=prompt,
                messages=full_messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                user=user,
                metadata=metadata
            )
            
            # Stream tokens
            full_response = ""
            token_count = 0
            start_time = datetime.utcnow()
            
            async for token in token_generator:
                full_response += token
                token_count += 1
                yield token
            
            # After streaming completes, save to history
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Calculate token usage
            prompt_tokens = self.token_counter.count_tokens(prompt)
            completion_tokens = self.token_counter.count_tokens(full_response)
            total_tokens = prompt_tokens + completion_tokens
            
            # Calculate cost
            cost = self.cost_calculator.calculate_cost(
                model=model or "unknown",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
            # Save to history
            await self._save_to_history(
                conversation_id=conversation_id,
                user_id=user,
                messages=messages,
                response={
                    "content": full_response,
                    "model": model or "auto-routed",
                    "tokens": total_tokens
                },
                model=model or "auto-routed",
                tokens=total_tokens,
                latency_ms=latency_ms,
                cache_hit=False,
                metadata={
                    "request_id": request_id,
                    "streaming": True,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tokens_per_second": token_count / (latency_ms / 1000) if latency_ms > 0 else 0,
                    "cost_usd": cost
                }
            )
            
            logger.info(
                "chat_stream_completed",
                request_id=request_id,
                conversation_id=conversation_id,
                model=model,
                tokens=total_tokens,
                latency_ms=round(latency_ms, 2),
                tokens_per_second=round(token_count / (latency_ms / 1000), 2) if latency_ms > 0 else 0
            )
            
        except Exception as e:
            logger.error(
                "chat_stream_failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    # ========================================================================
    # CONVERSATION MANAGEMENT
    # ========================================================================
    
    def _generate_conversation_id(self, user: Optional[str] = None) -> str:
        """Generate a unique conversation ID."""
        prefix = f"conv-{user}-" if user else "conv-"
        return f"{prefix}{uuid.uuid4().hex[:12]}"
    
    async def _get_conversation_context(
        self,
        conversation_id: str,
        user: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get conversation context from cache or database."""
        # Check active conversations cache
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        
        # Load from database
        history = await self.chat_repository.get_conversation_history(
            conversation_id=conversation_id,
            limit=self.max_context_messages
        )
        
        conversation = {
            "id": conversation_id,
            "user_id": user,
            "messages": history,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "token_count": sum(msg.get("tokens", 0) for msg in history),
            "message_count": len(history)
        }
        
        # Cache conversation
        self.active_conversations[conversation_id] = conversation
        
        return conversation
    
    async def _update_conversation_context(
        self,
        conversation_id: str,
        user: Optional[str],
        messages: List[Dict[str, str]],
        response: Dict[str, Any]
    ):
        """Update conversation context with new messages."""
        if conversation_id not in self.active_conversations:
            await self._get_conversation_context(conversation_id, user)
        
        conversation = self.active_conversations[conversation_id]
        
        # Add user messages
        for msg in messages:
            conversation["messages"].append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Add assistant response
        conversation["messages"].append({
            "role": "assistant",
            "content": response["content"],
            "model": response["model"],
            "tokens": response["tokens"],
            "timestamp": datetime.utcnow().isoformat()
        })
        
        conversation["token_count"] += response["tokens"]
        conversation["message_count"] += len(messages) + 1
        conversation["updated_at"] = datetime.utcnow()
        
        # Trim context if too long
        if conversation["token_count"] > self.max_context_tokens:
            await self._trim_conversation_context(conversation_id)
    
    async def _trim_conversation_context(self, conversation_id: str):
        """Trim conversation context to fit token limit."""
        if conversation_id not in self.active_conversations:
            return
        
        conversation = self.active_conversations[conversation_id]
        messages = conversation["messages"]
        
        # Keep system message if present
        system_messages = [m for m in messages if m["role"] == "system"]
        other_messages = [m for m in messages if m["role"] != "system"]
        
        # Remove oldest messages until under limit
        while conversation["token_count"] > self.max_context_tokens and len(other_messages) > 1:
            removed = other_messages.pop(0)
            conversation["token_count"] -= removed.get("tokens", 0)
        
        conversation["messages"] = system_messages + other_messages
    
    async def _prepare_conversation_context(
        self,
        conversation: Dict[str, Any],
        new_messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepare full conversation context for inference."""
        context = []
        
        # Add previous messages from conversation
        for msg in conversation.get("messages", []):
            context.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add new messages
        context.extend(new_messages)
        
        return context
    
    def _extract_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Extract the latest user prompt from messages."""
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            raise InvalidPromptError("No user message found in conversation")
        return user_messages[-1]["content"]
    
    # ========================================================================
    # CHAT HISTORY
    # ========================================================================
    
    async def _save_to_history(
        self,
        conversation_id: str,
        user_id: Optional[str],
        messages: List[Dict[str, str]],
        response: Dict[str, Any],
        model: str,
        tokens: int,
        latency_ms: float,
        cache_hit: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save chat interaction to history."""
        # Save user messages
        for msg in messages:
            await self.chat_repository.create_chat_entry({
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": msg["role"],
                "content": msg["content"],
                "model": None,
                "tokens": self.token_counter.count_tokens(msg["content"]),
                "latency_ms": 0,
                "metadata": metadata
            })
        
        # Save assistant response
        await self.chat_repository.create_chat_entry({
            "conversation_id": conversation_id,
            "user_id": user_id,
            "role": "assistant",
            "content": response["content"],
            "model": model,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "metadata": {
                **(metadata or {}),
                "cache_hit": cache_hit,
                "response_id": response.get("id")
            }
        })
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get conversation history."""
        # Verify access
        if user_id:
            conversation = await self.chat_repository.get_conversation(conversation_id)
            if conversation and conversation.get("user_id") != user_id:
                raise PermissionError("Access denied to this conversation")
        
        history = await self.chat_repository.get_conversation_history(
            conversation_id=conversation_id,
            limit=limit,
            offset=offset
        )
        
        total = await self.chat_repository.get_conversation_count(conversation_id)
        
        # Calculate conversation stats
        total_tokens = sum(entry.get("tokens", 0) for entry in history)
        total_cost = sum(entry.get("metadata", {}).get("cost_usd", 0) for entry in history)
        
        return {
            "conversation_id": conversation_id,
            "entries": history,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": total > (offset + limit),
            "stats": {
                "message_count": len(history),
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 6),
                "avg_tokens_per_message": total_tokens / len(history) if history else 0
            }
        }
    
    async def list_conversations(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List user's conversations."""
        conversations = await self.chat_repository.list_user_conversations(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        total = await self.chat_repository.get_user_conversation_count(user_id)
        
        return {
            "conversations": conversations,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": total > (offset + limit)
        }
    
    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Delete a conversation."""
        # Verify access
        if user_id:
            conversation = await self.chat_repository.get_conversation(conversation_id)
            if conversation and conversation.get("user_id") != user_id:
                raise PermissionError("Access denied to this conversation")
        
        # Remove from cache
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
        
        # Delete from database
        return await self.chat_repository.delete_conversation(conversation_id)
    
    # ========================================================================
    # FEEDBACK
    # ========================================================================
    
    async def submit_feedback(
        self,
        completion_id: str,
        rating: int,
        appropriate_model: bool,
        better_model: Optional[str] = None,
        comments: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit user feedback for a chat completion.
        
        Args:
            completion_id: ID of the completion
            rating: Rating from 1-5
            appropriate_model: Whether the model was appropriate
            better_model: Suggested better model
            comments: Additional comments
            tags: Feedback tags
            user_id: User identifier
        
        Returns:
            Feedback submission response
        """
        logger.info(
            "feedback_submitted",
            completion_id=completion_id,
            rating=rating,
            appropriate_model=appropriate_model,
            user_id=user_id
        )
        
        # Find the completion in history
        completion = await self.chat_repository.get_completion(completion_id)
        
        if not completion:
            raise ValueError(f"Completion {completion_id} not found")
        
        # Store feedback
        feedback = {
            "completion_id": completion_id,
            "user_id": user_id or completion.get("user_id"),
            "rating": rating,
            "appropriate_model": appropriate_model,
            "better_model": better_model,
            "comments": comments,
            "tags": tags or [],
            "created_at": datetime.utcnow().isoformat()
        }
        
        await self.chat_repository.save_feedback(feedback)
        
        # Update model performance with feedback
        if completion.get("model"):
            await self.routing_service.update_model_performance(
                model_id=completion["model"],
                latency_ms=completion.get("latency_ms", 0),
                tokens=completion.get("tokens", 0),
                cost=completion.get("metadata", {}).get("cost_usd", 0),
                success=rating >= 4  # Consider 4-5 as success
            )
        
        # If better model suggested, record for analysis
        if better_model:
            logger.info(
                "better_model_suggested",
                completion_id=completion_id,
                original_model=completion.get("model"),
                suggested_model=better_model
            )
        
        return {
            "status": "success",
            "message": "Feedback received. Thank you!",
            "feedback_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # ========================================================================
    # CONVERSATION SUMMARIZATION
    # ========================================================================
    
    async def summarize_conversation(
        self,
        conversation_id: str,
        max_length: int = 200
    ) -> str:
        """Generate a summary of a conversation."""
        # Get conversation history
        history = await self.chat_repository.get_conversation_history(
            conversation_id=conversation_id,
            limit=100
        )
        
        if not history:
            return "No messages in conversation"
        
        # Extract messages for summarization
        messages = [
            f"{msg['role']}: {msg['content'][:200]}"
            for msg in history[-20:]  # Last 20 messages
        ]
        
        conversation_text = "\n".join(messages)
        
        # Create summarization prompt
        summarization_prompt = f"""
        Summarize the following conversation in {max_length} characters or less:
        
        {conversation_text}
        
        Summary:
        """
        
        try:
            # Use a fast model for summarization
            response = await self.gateway_handler.process_request(
                request_id=str(uuid.uuid4()),
                prompt=summarization_prompt,
                model="tinyllama-1.1b",  # Fast, lightweight model
                max_tokens=100,
                temperature=0.3
            )
            
            summary = response["content"].strip()
            
            # Truncate if needed
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            return "Summary unavailable"
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    async def _update_statistics(self, response: Dict[str, Any]):
        """Update service statistics."""
        self.stats["total_tokens"] += response["tokens"]
        self.stats["total_cost"] += response.get("cost_usd", 0)
        
        # Update averages
        total = self.stats["total_chats"]
        self.stats["avg_latency_ms"] = (
            (self.stats["avg_latency_ms"] * (total - 1) + response["latency_ms"]) / total
        )
        self.stats["avg_tokens_per_message"] = (
            (self.stats["avg_tokens_per_message"] * (total - 1) + response["tokens"]) / total
        )
    
    async def _record_metrics(self, response: Dict[str, Any]):
        """Record chat metrics."""
        await self.metrics_collector.record_chat_completion(
            model=response["model"],
            tokens=response["tokens"],
            latency_ms=response["latency_ms"],
            cache_hit=response.get("cache_hit", False)
        )
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get chat service statistics."""
        return {
            **self.stats,
            "active_conversations": len(self.active_conversations),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def reset_statistics(self):
        """Reset chat service statistics."""
        self.stats = {
            "total_chats": 0,
            "total_messages": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_latency_ms": 0.0,
            "avg_tokens_per_message": 0
        }
        logger.info("chat_statistics_reset")
    
    def generate_cache_key(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate cache key for chat request."""
        # Extract the last user message for caching
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            return None
        
        last_prompt = user_messages[-1]["content"]
        
        key_data = {
            "prompt": last_prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()
        
        return f"chat:{key_hash}"


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_chat_service = None


def get_chat_service() -> ChatService:
    """Get singleton chat service instance."""
    global _chat_service
    if not _chat_service:
        _chat_service = ChatService()
    return _chat_service


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ChatService",
    "get_chat_service"
]