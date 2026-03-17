from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.schemas.routing import RoutingAnalyzeRequest, RoutingAnalyzeResponse
from backend.app.services.huggingface_client import HuggingFaceClient
from backend.app.services.groq_client import GroqClient
from backend.app.persistence.models import InferenceLog
from backend.app.cache.redis import get_redis
import time
import hashlib
import uuid
import json


class RoutingService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.hf_client = HuggingFaceClient()
        self.groq_client = GroqClient()

    async def analyze_prompt(self, request: RoutingAnalyzeRequest) -> RoutingAnalyzeResponse:
        redis_client = await get_redis()
        
        # 1. Routing Logic (Simple heuristical routing based on prompt length)
        # In a real enterprise system this would use a semantic classifier
        request_length = len(request.prompt)
        
        if request.preferred_model:
            selected_model = request.preferred_model
            strategy_used = "user_override"
        elif request_length > 400 or "code" in request.prompt.lower():
            # Heavier or code-related task -> route to capable/code models
            selected_model = "mistralai/Mistral-7B-Instruct-v0.2"
            strategy_used = "complexity_router (capable)"
        elif request_length < 120:
            # Short/simple prompts -> super fast Groq Llama 3.1
            selected_model = "groq/llama-3.1-8b-instant"
            strategy_used = "complexity_router (fast_groq)"
        else:
            # Default to a small, cheap HF model
            selected_model = "microsoft/phi-2"
            strategy_used = "complexity_router (phi2_mid)"

        alternative_candidates = [
            "groq/llama-3.1-8b-instant",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta",
        ]
        
        # 2. Semantic Caching Preparation
        cache_key = f"prompt_cache:{selected_model}:{hashlib.md5(request.prompt.encode()).hexdigest()}"
        cached_result = await redis_client.get(cache_key)

        # If the cache contains a previous "mock" response (from when keys weren't configured),
        # ignore it so that once keys are added the system starts returning real completions.
        if cached_result and (
            "Set GROQ_API_KEY in .env" in cached_result
            or "Inference Simulated:" in cached_result
            or "[Mock Groq" in cached_result
        ):
            cached_result = None
        
        start_time = time.time()
        latency_ms = 0.0
        response_text = ""
        status_flag = "success"
        error_msg = None
        cache_hit_flag = 0
        
        if cached_result:
            # 3. Cache Hit
            cache_hit_flag = 1
            response_text = cached_result
            latency_ms = (time.time() - start_time) * 1000
            strategy_used += " [CACHE HIT]"
        else:
            # 4. Actual Inference via appropriate provider
            if selected_model.startswith("groq/"):
                groq_model = selected_model.split("/", 1)[1]
                messages_payload = [
                    {"role": "user", "content": request.prompt}
                ]
                result = await self.groq_client.chat_completion(
                    model=groq_model,
                    messages=messages_payload,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            else:
                result = await self.hf_client.generate_text(
                    model_id=selected_model,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if result["status"] == "success":
                response_text = result["text"]
                # Store in cache for 1 hour (skip caching mock placeholders)
                if (
                    "Set GROQ_API_KEY in .env" not in response_text
                    and "Inference Simulated:" not in response_text
                    and "[Mock Groq" not in response_text
                ):
                    await redis_client.setex(cache_key, 3600, response_text)
            else:
                status_flag = "error"
                error_msg = result.get("error", "Unknown error")
                if "details" in result:
                    response_text = f"Inference Error: {error_msg}\n\nDetails:\n{result['details']}"
                else:
                    response_text = f"Inference Error: {error_msg}"
                
        # 5. DB Logging for Observability / Learning
        req_id = str(uuid.uuid4())
        
        log_entry = InferenceLog(
            request_id=req_id,
            prompt=request.prompt,
            selected_model=selected_model,
            strategy_used=strategy_used,
            response_text=response_text,
            latency_ms=latency_ms,
            cost_estimate=latency_ms * 0.00001, # fake cost based on time
            status=status_flag,
            error_message=error_msg,
            cache_hit=cache_hit_flag
        )
        self.db.add(log_entry)
        await self.db.commit()

        # 6. Response Construction
        return RoutingAnalyzeResponse(
             selected_model=selected_model,
             alternative_candidates=alternative_candidates,
             strategy_used=strategy_used,
             estimated_cost=log_entry.cost_estimate,
             estimated_latency=latency_ms,
             response_text=response_text
        )
