import uuid
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, AsyncIterator

from backend.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatResponseChoice,
    ChatMessage,
)
from backend.app.services.huggingface_client import HuggingFaceClient


import time
import json
from backend.app.persistence.models import InferenceLog

class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.hf_client = HuggingFaceClient()

    async def generate_response(
        self,
        request: ChatRequest,
        model_name: str,
        user=None,
    ) -> ChatResponse:
        start_time = time.time()
        
        # Flatten messages into a single prompt string
        prompt = self._build_prompt(request.messages)

        result = await self.hf_client.generate_text(
            model_id=model_name,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        latency_ms = (time.time() - start_time) * 1000
        response_text = result.get("text", "") if result["status"] == "success" else "Error generating response."
        
        # Log to DB for analytics
        log_entry = InferenceLog(
            request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            prompt=prompt,
            selected_model=model_name,
            strategy_used="chat_completion",
            response_text=response_text,
            latency_ms=latency_ms,
            cost_estimate=latency_ms * 0.00001, # placeholder calc
            status=result["status"],
            error_message=result.get("error") if result["status"] == "error" else None,
            cache_hit=0
        )
        self.db.add(log_entry)
        await self.db.commit()

        return ChatResponse(
            id=log_entry.request_id,
            model=model_name,
            choices=[
                ChatResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(response_text) // 4,
                "total_tokens": (len(prompt) + len(response_text)) // 4,
            },
        )

    async def stream_response(
        self,
        request: ChatRequest,
        model_name: str,
        user=None,
    ) -> AsyncIterator[str]:
        """SSE stream: yields chunks in OpenAI-compatible format."""
        prompt = self._build_prompt(request.messages)
        result = await self.hf_client.generate_text(
            model_id=model_name,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        text = result.get("text", "") if result["status"] == "success" else "Error."

        # Simulate streaming by chunking the response
        words = text.split()
        for i, word in enumerate(words):
            chunk = {
                "id": f"chatcmpl-stream",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": word + " "}, "index": 0}],
            }
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.03)  # Simulate streaming delay

        yield "data: [DONE]\n\n"

    @staticmethod
    def _build_prompt(messages: list[ChatMessage]) -> str:
        """Convert messages list to a single text prompt."""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"[System]: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        parts.append("Assistant:")
        return "\n".join(parts)
