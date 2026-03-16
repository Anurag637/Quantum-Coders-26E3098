from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.schemas.routing import RoutingAnalyzeRequest, RoutingAnalyzeResponse
from backend.app.services.huggingface_client import HuggingFaceClient
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

    async def analyze_prompt(self, request: RoutingAnalyzeRequest) -> RoutingAnalyzeResponse:
        redis_client = await get_redis()
        
        # 1. Routing Logic (Simple heuristical routing based on prompt length)
        # In a real enterprise system this would use a semantic classifier
        request_length = len(request.prompt)
        
        if request.preferred_model:
            selected_model = request.preferred_model
            strategy_used = "user_override"
        elif request_length > 200 or "code" in request.prompt.lower() or "explain" in request.prompt.lower():
            # Complex task -> Capable model
            selected_model = "mistralai/Mistral-7B-Instruct-v0.2"
            strategy_used = "complexity_router (capable)"
        else:
            # Simple task -> Fast/cheap model
            selected_model = "google/gemma-7b-it"
            strategy_used = "complexity_router (fast)"
            
        alternative_candidates = ["meta-llama/Llama-2-7b-chat-hf"] if "mistral" in selected_model else ["mistralai/Mistral-7B-Instruct-v0.2"]
        
        # 2. Semantic Caching Preparation
        cache_key = f"prompt_cache:{selected_model}:{hashlib.md5(request.prompt.encode()).hexdigest()}"
        cached_result = await redis_client.get(cache_key)
        
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
            # 4. Actual Inference via Hugging Face Gateway
            result = await self.hf_client.generate_text(
                model_id=selected_model,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if result["status"] == "success":
                response_text = result["text"]
                # Store in cache for 1 hour
                await redis_client.setex(cache_key, 3600, response_text)
            else:
                status_flag = "error"
                response_text = "Inference Error: model may be loading or unavailable."
                error_msg = result.get("error", "Unknown error")
                
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
