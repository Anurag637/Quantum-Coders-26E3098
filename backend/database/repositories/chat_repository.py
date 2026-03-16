"""
Chat Repository - Production Ready
Database operations for chat history, conversations, messages, and analytics.
Provides CRUD operations with pagination, filtering, and aggregation.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import json

from sqlalchemy import select, update, delete, and_, or_, desc, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.logging import get_logger
from database.models import ChatHistory, User, ModelRegistry
from database.repositories.base import BaseRepository
from core.exceptions import ResourceNotFoundError

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# CHAT REPOSITORY
# ============================================================================

class ChatRepository(BaseRepository[ChatHistory]):
    """
    Repository for chat history operations.
    
    Features:
    - Create/Read/Delete chat messages
    - Conversation management
    - Pagination and filtering
    - Usage statistics
    - Analytics queries
    - Bulk operations
    """
    
    def __init__(self, session: Optional[AsyncSession] = None):
        super().__init__(ChatHistory, session)
    
    # ========================================================================
    # CREATE OPERATIONS
    # ========================================================================
    
    async def create_message(
        self,
        user_id: Optional[UUID],
        conversation_id: str,
        role: str,
        content: str,
        model_id: Optional[str] = None,
        tokens: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatHistory:
        """
        Create a new chat message.
        
        Args:
            user_id: User ID (None for anonymous)
            conversation_id: Unique conversation identifier
            role: Message role (user, assistant, system)
            content: Message content
            model_id: Model used for response
            tokens: Total tokens used
            prompt_tokens: Prompt tokens
            completion_tokens: Completion tokens
            latency_ms: Inference latency
            cost: Cost in USD
            metadata: Additional metadata
        
        Returns:
            Created chat message
        """
        message = ChatHistory(
            user_id=user_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            model_id=model_id,
            tokens_used=tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost=cost,
            metadata=metadata or {}
        )
        
        self.session.add(message)
        await self.session.flush()
        await self.session.refresh(message)
        
        logger.debug(
            "chat_message_created",
            message_id=str(message.id),
            conversation_id=conversation_id,
            role=role,
            model_id=model_id,
            user_id=str(user_id) if user_id else None
        )
        
        return message
    
    async def create_conversation(
        self,
        user_id: Optional[UUID],
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new conversation.
        
        Args:
            user_id: User ID
            title: Conversation title
            metadata: Additional metadata
        
        Returns:
            Conversation ID
        """
        import uuid
        conversation_id = str(uuid.uuid4())
        
        # Create system message to initialize conversation
        await self.create_message(
            user_id=user_id,
            conversation_id=conversation_id,
            role="system",
            content=title or "New conversation",
            metadata={"type": "conversation_start", **metadata} if metadata else {"type": "conversation_start"}
        )
        
        logger.info(
            "conversation_created",
            conversation_id=conversation_id,
            user_id=str(user_id) if user_id else None,
            title=title
        )
        
        return conversation_id
    
    # ========================================================================
    # READ OPERATIONS
    # ========================================================================
    
    async def get_conversation(
        self,
        conversation_id: str,
        user_id: Optional[UUID] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[ChatHistory], int]:
        """
        Get all messages in a conversation.
        
        Args:
            conversation_id: Conversation ID
            user_id: Optional user ID for authorization
            limit: Maximum messages to return
            offset: Pagination offset
        
        Returns:
            Tuple of (messages, total_count)
        """
        query = select(ChatHistory).where(
            ChatHistory.conversation_id == conversation_id
        )
        
        if user_id:
            query = query.where(ChatHistory.user_id == user_id)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.session.execute(count_query)
        total_count = total.scalar_one()
        
        # Get paginated messages
        query = query.order_by(ChatHistory.created_at.asc())
        query = query.offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        messages = result.scalars().all()
        
        return messages, total_count
    
    async def get_conversations(
        self,
        user_id: Optional[UUID],
        limit: int = 50,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        model_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get all conversations for a user.
        
        Args:
            user_id: User ID (None for anonymous)
            limit: Maximum conversations to return
            offset: Pagination offset
            start_date: Filter by start date
            end_date: Filter by end date
            model_id: Filter by model used
        
        Returns:
            Tuple of (conversation_summaries, total_count)
        """
        # Subquery to get latest message per conversation
        subquery = select(
            ChatHistory.conversation_id,
            func.max(ChatHistory.created_at).label('last_message_at'),
            func.count(ChatHistory.id).label('message_count'),
            func.sum(ChatHistory.tokens_used).label('total_tokens'),
            func.sum(ChatHistory.cost).label('total_cost')
        ).group_by(ChatHistory.conversation_id)
        
        if user_id:
            subquery = subquery.where(ChatHistory.user_id == user_id)
        if model_id:
            subquery = subquery.where(ChatHistory.model_id == model_id)
        if start_date:
            subquery = subquery.where(ChatHistory.created_at >= start_date)
        if end_date:
            subquery = subquery.where(ChatHistory.created_at <= end_date)
        
        subquery = subquery.subquery()
        
        # Get first message (usually the user's first message) as title
        title_query = select(
            ChatHistory.conversation_id,
            ChatHistory.content.label('title'),
            ChatHistory.created_at.label('created_at')
        ).where(
            ChatHistory.role == 'user'
        ).distinct(ChatHistory.conversation_id).order_by(
            ChatHistory.conversation_id,
            ChatHistory.created_at.asc()
        ).subquery()
        
        # Main query
        query = select(
            subquery.c.conversation_id,
            subquery.c.last_message_at,
            subquery.c.message_count,
            subquery.c.total_tokens,
            subquery.c.total_cost,
            title_query.c.title,
            title_query.c.created_at
        ).join(
            title_query,
            subquery.c.conversation_id == title_query.c.conversation_id,
            isouter=True
        ).order_by(desc(subquery.c.last_message_at))
        
        # Get total count
        count_query = select(func.count()).select_from(subquery)
        total = await self.session.execute(count_query)
        total_count = total.scalar_one()
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        rows = result.all()
        
        conversations = []
        for row in rows:
            conversations.append({
                "conversation_id": row.conversation_id,
                "title": row.title or "New conversation",
                "created_at": row.created_at or row.last_message_at,
                "last_message_at": row.last_message_at,
                "message_count": row.message_count,
                "total_tokens": row.total_tokens or 0,
                "total_cost": float(row.total_cost) if row.total_cost else 0.0
            })
        
        return conversations, total_count
    
    async def get_message(
        self,
        message_id: UUID,
        user_id: Optional[UUID] = None
    ) -> Optional[ChatHistory]:
        """
        Get a specific message by ID.
        
        Args:
            message_id: Message UUID
            user_id: Optional user ID for authorization
        
        Returns:
            Chat message or None
        """
        query = select(ChatHistory).where(ChatHistory.id == message_id)
        
        if user_id:
            query = query.where(ChatHistory.user_id == user_id)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def search_messages(
        self,
        query_text: str,
        user_id: Optional[UUID] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[ChatHistory], int]:
        """
        Search messages by content.
        
        Args:
            query_text: Search query
            user_id: Optional user ID filter
            limit: Maximum results
            offset: Pagination offset
        
        Returns:
            Tuple of (matching_messages, total_count)
        """
        # PostgreSQL full-text search
        search_query = func.to_tsquery('english', query_text.replace(' ', ' & '))
        
        query = select(ChatHistory).where(
            func.to_tsvector('english', ChatHistory.content).op('@@')(search_query)
        )
        
        if user_id:
            query = query.where(ChatHistory.user_id == user_id)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.session.execute(count_query)
        total_count = total.scalar_one()
        
        # Get paginated results
        query = query.order_by(desc(ChatHistory.created_at))
        query = query.offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        messages = result.scalars().all()
        
        return messages, total_count
    
    # ========================================================================
    # UPDATE OPERATIONS
    # ========================================================================
    
    async def update_message(
        self,
        message_id: UUID,
        user_id: Optional[UUID],
        **updates
    ) -> Optional[ChatHistory]:
        """
        Update a message.
        
        Args:
            message_id: Message UUID
            user_id: User ID for authorization
            **updates: Fields to update
        
        Returns:
            Updated message or None
        """
        query = select(ChatHistory).where(
            and_(
                ChatHistory.id == message_id,
                ChatHistory.user_id == user_id
            )
        )
        
        result = await self.session.execute(query)
        message = result.scalar_one_or_none()
        
        if not message:
            raise ResourceNotFoundError(
                resource_type="ChatMessage",
                resource_id=str(message_id)
            )
        
        for key, value in updates.items():
            if hasattr(message, key):
                setattr(message, key, value)
        
        await self.session.flush()
        await self.session.refresh(message)
        
        logger.debug(
            "chat_message_updated",
            message_id=str(message_id),
            updates=list(updates.keys())
        )
        
        return message
    
    async def rename_conversation(
        self,
        conversation_id: str,
        user_id: UUID,
        new_title: str
    ) -> bool:
        """
        Rename a conversation.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID
            new_title: New conversation title
        
        Returns:
            True if successful
        """
        # Find the system message that contains the title
        query = select(ChatHistory).where(
            and_(
                ChatHistory.conversation_id == conversation_id,
                ChatHistory.user_id == user_id,
                ChatHistory.role == 'system',
                ChatHistory.metadata['type'].astext == 'conversation_start'
            )
        )
        
        result = await self.session.execute(query)
        message = result.scalar_one_or_none()
        
        if not message:
            # Create new system message with title
            await self.create_message(
                user_id=user_id,
                conversation_id=conversation_id,
                role="system",
                content=new_title,
                metadata={"type": "conversation_start", "renamed": True}
            )
        else:
            # Update existing title
            message.content = new_title
            await self.session.flush()
        
        logger.info(
            "conversation_renamed",
            conversation_id=conversation_id,
            user_id=str(user_id),
            new_title=new_title
        )
        
        return True
    
    # ========================================================================
    # DELETE OPERATIONS
    # ========================================================================
    
    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: UUID
    ) -> bool:
        """
        Delete an entire conversation.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID for authorization
        
        Returns:
            True if deleted
        """
        query = delete(ChatHistory).where(
            and_(
                ChatHistory.conversation_id == conversation_id,
                ChatHistory.user_id == user_id
            )
        )
        
        result = await self.session.execute(query)
        await self.session.flush()
        
        if result.rowcount > 0:
            logger.info(
                "conversation_deleted",
                conversation_id=conversation_id,
                user_id=str(user_id),
                messages_deleted=result.rowcount
            )
            return True
        
        return False
    
    async def delete_message(
        self,
        message_id: UUID,
        user_id: UUID
    ) -> bool:
        """
        Delete a single message.
        
        Args:
            message_id: Message UUID
            user_id: User ID for authorization
        
        Returns:
            True if deleted
        """
        query = delete(ChatHistory).where(
            and_(
                ChatHistory.id == message_id,
                ChatHistory.user_id == user_id
            )
        )
        
        result = await self.session.execute(query)
        await self.session.flush()
        
        if result.rowcount > 0:
            logger.debug(
                "chat_message_deleted",
                message_id=str(message_id),
                user_id=str(user_id)
            )
            return True
        
        return False
    
    async def delete_user_conversations(
        self,
        user_id: UUID,
        older_than: Optional[datetime] = None
    ) -> int:
        """
        Delete all conversations for a user.
        
        Args:
            user_id: User ID
            older_than: Optional age filter
        
        Returns:
            Number of messages deleted
        """
        query = delete(ChatHistory).where(ChatHistory.user_id == user_id)
        
        if older_than:
            query = query.where(ChatHistory.created_at < older_than)
        
        result = await self.session.execute(query)
        await self.session.flush()
        
        logger.info(
            "user_conversations_deleted",
            user_id=str(user_id),
            messages_deleted=result.rowcount,
            older_than=older_than.isoformat() if older_than else None
        )
        
        return result.rowcount
    
    # ========================================================================
    # ANALYTICS & STATISTICS
    # ========================================================================
    
    async def get_user_usage_stats(
        self,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get chat usage statistics for a user.
        
        Args:
            user_id: User ID
            start_date: Start date for aggregation
            end_date: End date for aggregation
        
        Returns:
            Dictionary with usage statistics
        """
        query = select(
            func.count(ChatHistory.id).label('total_messages'),
            func.count(func.distinct(ChatHistory.conversation_id)).label('total_conversations'),
            func.sum(ChatHistory.tokens_used).label('total_tokens'),
            func.sum(ChatHistory.prompt_tokens).label('total_prompt_tokens'),
            func.sum(ChatHistory.completion_tokens).label('total_completion_tokens'),
            func.avg(ChatHistory.latency_ms).label('avg_latency_ms'),
            func.sum(ChatHistory.cost).label('total_cost'),
            func.max(ChatHistory.created_at).label('last_active')
        ).where(ChatHistory.user_id == user_id)
        
        if start_date:
            query = query.where(ChatHistory.created_at >= start_date)
        if end_date:
            query = query.where(ChatHistory.created_at <= end_date)
        
        result = await self.session.execute(query)
        row = result.first()
        
        # Get model usage breakdown
        model_query = select(
            ChatHistory.model_id,
            func.count().label('count'),
            func.sum(ChatHistory.tokens_used).label('tokens'),
            func.sum(ChatHistory.cost).label('cost')
        ).where(
            and_(
                ChatHistory.user_id == user_id,
                ChatHistory.model_id.isnot(None)
            )
        )
        
        if start_date:
            model_query = model_query.where(ChatHistory.created_at >= start_date)
        if end_date:
            model_query = model_query.where(ChatHistory.created_at <= end_date)
        
        model_query = model_query.group_by(ChatHistory.model_id)
        
        model_result = await self.session.execute(model_query)
        model_usage = []
        
        for m in model_result.all():
            model_usage.append({
                "model_id": m.model_id,
                "messages": m.count,
                "tokens": m.tokens or 0,
                "cost": float(m.cost) if m.cost else 0.0
            })
        
        # Get daily activity
        daily_query = select(
            func.date(ChatHistory.created_at).label('date'),
            func.count().label('messages'),
            func.count(func.distinct(ChatHistory.conversation_id)).label('conversations'),
            func.sum(ChatHistory.tokens_used).label('tokens')
        ).where(ChatHistory.user_id == user_id)
        
        if start_date:
            daily_query = daily_query.where(ChatHistory.created_at >= start_date)
        if end_date:
            daily_query = daily_query.where(ChatHistory.created_at <= end_date)
        
        daily_query = daily_query.group_by(func.date(ChatHistory.created_at))
        daily_query = daily_query.order_by(func.date(ChatHistory.created_at).desc())
        daily_query = daily_query.limit(30)
        
        daily_result = await self.session.execute(daily_query)
        daily_activity = []
        
        for d in daily_result.all():
            daily_activity.append({
                "date": d.date.isoformat(),
                "messages": d.messages,
                "conversations": d.conversations,
                "tokens": d.tokens or 0
            })
        
        return {
            "user_id": str(user_id),
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "total_messages": row.total_messages or 0,
            "total_conversations": row.total_conversations or 0,
            "total_tokens": row.total_tokens or 0,
            "total_prompt_tokens": row.total_prompt_tokens or 0,
            "total_completion_tokens": row.total_completion_tokens or 0,
            "avg_latency_ms": round(row.avg_latency_ms, 2) if row.avg_latency_ms else 0,
            "total_cost": float(row.total_cost) if row.total_cost else 0.0,
            "last_active": row.last_active.isoformat() if row.last_active else None,
            "model_usage": model_usage,
            "daily_activity": daily_activity
        }
    
    async def get_global_usage_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get global chat usage statistics.
        
        Args:
            start_date: Start date for aggregation
            end_date: End date for aggregation
        
        Returns:
            Dictionary with global statistics
        """
        query = select(
            func.count(ChatHistory.id).label('total_messages'),
            func.count(func.distinct(ChatHistory.conversation_id)).label('total_conversations'),
            func.count(func.distinct(ChatHistory.user_id)).label('total_users'),
            func.sum(ChatHistory.tokens_used).label('total_tokens'),
            func.sum(ChatHistory.cost).label('total_cost'),
            func.avg(ChatHistory.latency_ms).label('avg_latency_ms')
        )
        
        if start_date:
            query = query.where(ChatHistory.created_at >= start_date)
        if end_date:
            query = query.where(ChatHistory.created_at <= end_date)
        
        result = await self.session.execute(query)
        row = result.first()
        
        # Get model popularity
        model_query = select(
            ChatHistory.model_id,
            func.count().label('count'),
            func.sum(ChatHistory.tokens_used).label('tokens'),
            func.sum(ChatHistory.cost).label('cost')
        ).where(ChatHistory.model_id.isnot(None))
        
        if start_date:
            model_query = model_query.where(ChatHistory.created_at >= start_date)
        if end_date:
            model_query = model_query.where(ChatHistory.created_at <= end_date)
        
        model_query = model_query.group_by(ChatHistory.model_id)
        model_query = model_query.order_by(desc(func.count()))
        model_query = model_query.limit(10)
        
        model_result = await self.session.execute(model_query)
        top_models = []
        
        for m in model_result.all():
            top_models.append({
                "model_id": m.model_id,
                "messages": m.count,
                "tokens": m.tokens or 0,
                "cost": float(m.cost) if m.cost else 0.0,
                "percentage": 0.0  # Will calculate after
            })
        
        # Calculate percentages
        total_messages = row.total_messages or 0
        if total_messages > 0:
            for model in top_models:
                model["percentage"] = round(model["messages"] / total_messages * 100, 2)
        
        # Get hourly distribution
        hourly_query = select(
            func.extract('hour', ChatHistory.created_at).label('hour'),
            func.count().label('messages')
        )
        
        if start_date:
            hourly_query = hourly_query.where(ChatHistory.created_at >= start_date)
        if end_date:
            hourly_query = hourly_query.where(ChatHistory.created_at <= end_date)
        
        hourly_query = hourly_query.group_by(func.extract('hour', ChatHistory.created_at))
        hourly_query = hourly_query.order_by(func.extract('hour', ChatHistory.created_at))
        
        hourly_result = await self.session.execute(hourly_query)
        hourly_distribution = []
        
        for h in hourly_result.all():
            hourly_distribution.append({
                "hour": int(h.hour),
                "messages": h.messages
            })
        
        return {
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "total_messages": row.total_messages or 0,
            "total_conversations": row.total_conversations or 0,
            "total_users": row.total_users or 0,
            "total_tokens": row.total_tokens or 0,
            "total_cost": float(row.total_cost) if row.total_cost else 0.0,
            "avg_latency_ms": round(row.avg_latency_ms, 2) if row.avg_latency_ms else 0,
            "top_models": top_models,
            "hourly_distribution": hourly_distribution
        }
    
    async def get_conversation_summary(
        self,
        conversation_id: str,
        user_id: Optional[UUID] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get summary of a conversation.
        
        Args:
            conversation_id: Conversation ID
            user_id: Optional user ID for authorization
        
        Returns:
            Conversation summary or None
        """
        messages, total = await self.get_conversation(
            conversation_id,
            user_id,
            limit=1000,  # Get all messages for summary
            offset=0
        )
        
        if not messages:
            return None
        
        # Calculate statistics
        user_messages = [m for m in messages if m.role == 'user']
        assistant_messages = [m for m in messages if m.role == 'assistant']
        system_messages = [m for m in messages if m.role == 'system']
        
        total_tokens = sum(m.tokens_used or 0 for m in messages)
        total_cost = sum(m.cost or 0.0 for m in messages)
        avg_latency = sum(m.latency_ms or 0 for m in assistant_messages) / len(assistant_messages) if assistant_messages else 0
        
        # Get unique models used
        models_used = list(set(m.model_id for m in messages if m.model_id))
        
        # Get first user message as potential title
        title = None
        created_at = None
        for m in messages:
            if m.role == 'user':
                title = m.content[:100] + '...' if len(m.content) > 100 else m.content
                created_at = m.created_at
                break
        
        return {
            "conversation_id": conversation_id,
            "title": title or "Untitled conversation",
            "user_id": str(messages[0].user_id) if messages[0].user_id else None,
            "created_at": created_at.isoformat() if created_at else messages[0].created_at.isoformat(),
            "last_message_at": messages[-1].created_at.isoformat(),
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "system_messages": len(system_messages),
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
            "avg_latency_ms": round(avg_latency, 2),
            "models_used": models_used,
            "message_count": len(messages)
        }
    
    # ========================================================================
    # BULK OPERATIONS
    # ========================================================================
    
    async def bulk_insert_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> int:
        """
        Bulk insert multiple messages.
        
        Args:
            messages: List of message dictionaries
        
        Returns:
            Number of messages inserted
        """
        db_messages = []
        
        for msg in messages:
            db_messages.append(ChatHistory(
                user_id=msg.get('user_id'),
                conversation_id=msg['conversation_id'],
                role=msg['role'],
                content=msg['content'],
                model_id=msg.get('model_id'),
                tokens_used=msg.get('tokens'),
                prompt_tokens=msg.get('prompt_tokens'),
                completion_tokens=msg.get('completion_tokens'),
                latency_ms=msg.get('latency_ms'),
                cost=msg.get('cost'),
                metadata=msg.get('metadata', {})
            ))
        
        self.session.add_all(db_messages)
        await self.session.flush()
        
        logger.info(
            "bulk_messages_inserted",
            count=len(db_messages)
        )
        
        return len(db_messages)
    
    async def export_conversation(
        self,
        conversation_id: str,
        user_id: UUID,
        format: str = 'json'
    ) -> Dict[str, Any]:
        """
        Export a conversation in various formats.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID
            format: Export format (json, text, markdown)
        
        Returns:
            Exported conversation data
        """
        messages, _ = await self.get_conversation(
            conversation_id,
            user_id,
            limit=1000,
            offset=0
        )
        
        if not messages:
            raise ResourceNotFoundError(
                resource_type="Conversation",
                resource_id=conversation_id
            )
        
        summary = await self.get_conversation_summary(conversation_id, user_id)
        
        if format == 'json':
            return {
                "metadata": summary,
                "messages": [
                    {
                        "id": str(m.id),
                        "role": m.role,
                        "content": m.content,
                        "model": m.model_id,
                        "tokens": m.tokens_used,
                        "latency_ms": m.latency_ms,
                        "cost": float(m.cost) if m.cost else None,
                        "created_at": m.created_at.isoformat()
                    }
                    for m in messages
                ]
            }
        elif format == 'text':
            lines = []
            lines.append(f"Conversation: {summary['title']}")
            lines.append(f"Date: {summary['created_at']}")
            lines.append(f"Messages: {summary['total_messages']}")
            lines.append("=" * 50)
            lines.append("")
            
            for m in messages:
                lines.append(f"[{m.role.upper()}]")
                lines.append(m.content)
                lines.append("")
            
            return {"text": "\n".join(lines)}
        
        elif format == 'markdown':
            lines = []
            lines.append(f"# {summary['title']}")
            lines.append(f"")
            lines.append(f"- **Date**: {summary['created_at']}")
            lines.append(f"- **Messages**: {summary['total_messages']}")
            lines.append(f"- **Tokens**: {summary['total_tokens']}")
            lines.append(f"- **Cost**: ${summary['total_cost']:.6f}")
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")
            
            for m in messages:
                if m.role == 'user':
                    lines.append(f"## 👤 User")
                elif m.role == 'assistant':
                    lines.append(f"## 🤖 Assistant ({m.model_id or 'unknown'})")
                else:
                    lines.append(f"## ⚙️ System")
                
                lines.append(f"")
                lines.append(m.content)
                lines.append(f"")
                lines.append(f"---")
                lines.append(f"")
            
            return {"markdown": "\n".join(lines)}
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ChatRepository"
]