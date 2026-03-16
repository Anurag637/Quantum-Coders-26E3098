"""
Intelligent Prompt Routing Endpoints - Production Ready
Dynamic model selection based on prompt analysis, cost optimization, latency requirements,
and model capabilities. Implements multiple routing strategies with fallback chains.
"""

from fastapi import APIRouter, Request, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional, Union
import time
import uuid
import asyncio
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import hashlib
import json

from core.logging import get_logger
from core.security import verify_api_key, verify_admin, get_current_user
from core.rate_limiter import rate_limit
from core.exceptions import RoutingError, ModelNotAvailableError
from config import settings
from router.prompt_router import PromptRouter
from router.prompt_analyzer import PromptAnalyzer
from router.routing_strategies import RoutingStrategyFactory
from models.model_manager import ModelManager
from models.model_registry import ModelRegistry
from monitoring.metrics import MetricsCollector
from database.repositories.routing_repository import RoutingRepository
from cache.cache_manager import CacheManager

# Initialize router
router = APIRouter(prefix="/routing", tags=["Routing"])

# Initialize logger
logger = get_logger(__name__)

# Initialize services
prompt_router = PromptRouter()
prompt_analyzer = PromptAnalyzer()
strategy_factory = RoutingStrategyFactory()
model_manager = ModelManager()
model_registry = ModelRegistry()
metrics_collector = MetricsCollector()
routing_repository = RoutingRepository()
cache_manager = CacheManager()

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class RoutingRequest(BaseModel):
    """Routing analysis request schema"""
    prompt: str = Field(..., description="The prompt to analyze and route", min_length=1, max_length=10000)
    strategy: Optional[str] = Field(None, description="Routing strategy to use (latency, cost, quality, hybrid, auto)")
    preferred_models: Optional[List[str]] = Field(None, description="Preferred model IDs to consider")
    excluded_models: Optional[List[str]] = Field(None, description="Model IDs to exclude from routing")
    required_capabilities: Optional[List[str]] = Field(None, description="Required model capabilities")
    max_latency_ms: Optional[int] = Field(None, description="Maximum acceptable latency in ms", ge=50, le=10000)
    max_cost_usd: Optional[float] = Field(None, description="Maximum acceptable cost per request in USD", ge=0, le=1)
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for routing")
    
    @validator('strategy')
    def validate_strategy(cls, v):
        if v and v not in ['latency', 'cost', 'quality', 'hybrid', 'auto', None]:
            raise ValueError('Strategy must be latency, cost, quality, hybrid, or auto')
        return v


class RoutingDecision(BaseModel):
    """Routing decision schema"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Decision timestamp")
    prompt_hash: str = Field(..., description="Hash of the prompt")
    prompt_length: int = Field(..., description="Length of prompt in characters")
    prompt_tokens: Optional[int] = Field(None, description="Estimated token count")
    
    # Analysis results
    prompt_type: str = Field(..., description="Detected prompt type (code, qa, creative, reasoning, etc.)")
    complexity_score: float = Field(..., description="Prompt complexity score (0-1)")
    sentiment_score: Optional[float] = Field(None, description="Sentiment score (0-1)")
    language: str = Field("en", description="Detected language")
    entities: Optional[List[str]] = Field(None, description="Detected entities")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    
    # Routing decision
    selected_model: str = Field(..., description="Selected model ID")
    fallback_model: Optional[str] = Field(None, description="Fallback model ID")
    routing_strategy: str = Field(..., description="Strategy used for routing")
    confidence_score: float = Field(..., description="Confidence in this decision (0-1)")
    reasoning: str = Field(..., description="Human-readable explanation of the decision")
    
    # Performance metrics
    analysis_time_ms: float = Field(..., description="Time spent on analysis in ms")
    decision_time_ms: float = Field(..., description="Time spent on decision in ms")
    total_time_ms: float = Field(..., description="Total routing time in ms")
    
    # Alternatives considered
    alternatives: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative models considered")
    
    # Cache information
    cache_hit: bool = Field(False, description="Whether decision was cached")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchRoutingRequest(BaseModel):
    """Batch routing analysis request"""
    prompts: List[str] = Field(..., description="List of prompts to analyze", min_items=1, max_items=100)
    strategy: Optional[str] = Field(None, description="Routing strategy to use")
    parallel: bool = Field(True, description="Process prompts in parallel")


class BatchRoutingResponse(BaseModel):
    """Batch routing response schema"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    decisions: List[RoutingDecision] = Field(..., description="Routing decisions for each prompt")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    processing_time_ms: float = Field(..., description="Total processing time in ms")


class RoutingStrategyInfo(BaseModel):
    """Routing strategy information"""
    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    weights: Optional[Dict[str, float]] = Field(None, description="Strategy weights (if applicable)")
    use_cases: List[str] = Field(..., description="Recommended use cases")
    limitations: List[str] = Field(..., description="Known limitations")


class RoutingMetricsResponse(BaseModel):
    """Routing metrics response"""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    total_decisions: int = Field(..., description="Total routing decisions")
    average_confidence: float = Field(..., description="Average confidence score")
    average_analysis_time_ms: float = Field(..., description="Average analysis time")
    average_decision_time_ms: float = Field(..., description="Average decision time")
    cache_hit_rate: float = Field(..., description="Routing decision cache hit rate")
    
    # Model distribution
    model_distribution: Dict[str, int] = Field(..., description="Model selection counts")
    strategy_distribution: Dict[str, int] = Field(..., description="Strategy usage counts")
    prompt_type_distribution: Dict[str, int] = Field(..., description="Prompt type counts")
    
    # Performance by strategy
    strategy_performance: Dict[str, Dict[str, float]] = Field(..., description="Performance metrics by strategy")
    
    # Recent trends
    hourly_decisions: List[Dict[str, Any]] = Field(..., description="Hourly decision counts")
    hourly_latency: List[Dict[str, Any]] = Field(..., description="Hourly latency trends")


class RoutingRule(BaseModel):
    """Custom routing rule schema"""
    id: Optional[str] = Field(None, description="Rule ID")
    name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    conditions: List[Dict[str, Any]] = Field(..., description="Rule conditions")
    action: Dict[str, Any] = Field(..., description="Rule action (model selection)")
    priority: int = Field(0, description="Rule priority (higher = more important)")
    enabled: bool = Field(True, description="Whether rule is enabled")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# ============================================================================
# PROMPT ANALYSIS & ROUTING ENDPOINTS
# ============================================================================

@router.post(
    "/analyze",
    summary="Analyze Prompt",
    description="""
    Analyze a prompt without making a routing decision.
    
    Returns detailed prompt analysis including:
    - Prompt type classification
    - Complexity score
    - Language detection
    - Entity extraction
    - Keyword extraction
    - Sentiment analysis
    - Token estimation
    
    Useful for understanding prompts before routing.
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(rate_limit(limit=100, period=60))]
)
async def analyze_prompt(
    request: Request,
    analysis_request: RoutingRequest,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Analyze a prompt and return detailed characteristics.
    No routing decision is made - just pure analysis.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "prompt_analysis_requested",
        request_id=request_id,
        prompt_length=len(analysis_request.prompt),
        prompt_preview=analysis_request.prompt[:100] + "..." if len(analysis_request.prompt) > 100 else analysis_request.prompt
    )
    
    try:
        # Analyze the prompt
        analysis = await prompt_analyzer.analyze(analysis_request.prompt)
        
        # Add request metadata
        analysis["request_id"] = request_id
        analysis["analysis_time_ms"] = round((time.time() - start_time) * 1000, 2)
        analysis["timestamp"] = datetime.now().isoformat()
        
        # Add token estimation if not present
        if "prompt_tokens" not in analysis:
            from utils.token_counter import estimate_tokens
            analysis["prompt_tokens"] = estimate_tokens(analysis_request.prompt)
        
        logger.info(
            "prompt_analysis_completed",
            request_id=request_id,
            prompt_type=analysis.get("prompt_type", "unknown"),
            complexity=analysis.get("complexity_score", 0),
            analysis_time_ms=analysis["analysis_time_ms"]
        )
        
        return analysis
        
    except Exception as e:
        logger.error(
            "prompt_analysis_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "AnalysisFailed",
                "message": "Failed to analyze prompt",
                "request_id": request_id
            }
        )


@router.post(
    "/decide",
    summary="Make Routing Decision",
    description="""
    Make an intelligent routing decision for a prompt.
    
    This endpoint:
    1. Analyzes the prompt characteristics
    2. Considers available models and their capabilities
    3. Applies the selected routing strategy
    4. Returns the optimal model selection
    5. Provides reasoning for the decision
    
    Supports custom strategies, preferred/excluded models,
    and latency/cost constraints.
    
    Rate limited: 50 requests per minute
    """,
    response_model=RoutingDecision,
    dependencies=[Depends(rate_limit(limit=50, period=60))]
)
async def make_routing_decision(
    request: Request,
    routing_request: RoutingRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> RoutingDecision:
    """
    Make an intelligent routing decision for a prompt.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "routing_decision_requested",
        request_id=request_id,
        prompt_length=len(routing_request.prompt),
        strategy=routing_request.strategy,
        preferred_models=routing_request.preferred_models
    )
    
    try:
        # Generate cache key for this request
        cache_key = _generate_cache_key(routing_request)
        
        # Check cache first (if enabled)
        cached_decision = None
        if settings.cache.enabled:
            cached_decision = await cache_manager.get(cache_key)
        
        if cached_decision:
            logger.info(
                "routing_cache_hit",
                request_id=request_id,
                cache_key=cache_key
            )
            
            # Record cache hit metric
            background_tasks.add_task(
                metrics_collector.record_routing_cache_hit,
                strategy=cached_decision.get("routing_strategy", "unknown")
            )
            
            # Update timestamp and request ID
            cached_decision["request_id"] = request_id
            cached_decision["timestamp"] = datetime.now()
            cached_decision["cache_hit"] = True
            cached_decision["total_time_ms"] = round((time.time() - start_time) * 1000, 2)
            
            return RoutingDecision(**cached_decision)
        
        # No cache hit - perform full routing
        logger.info(
            "routing_cache_miss",
            request_id=request_id,
            cache_key=cache_key
        )
        
        # Step 1: Analyze the prompt
        analysis_start = time.time()
        analysis = await prompt_analyzer.analyze(routing_request.prompt)
        analysis_time = (time.time() - analysis_start) * 1000
        
        # Step 2: Get available models
        models = await model_manager.get_available_models()
        
        # Filter models based on request
        if routing_request.preferred_models:
            models = [m for m in models if m["id"] in routing_request.preferred_models]
        
        if routing_request.excluded_models:
            models = [m for m in models if m["id"] not in routing_request.excluded_models]
        
        if routing_request.required_capabilities:
            models = [
                m for m in models 
                if all(cap in m.get("capabilities", []) for cap in routing_request.required_capabilities)
            ]
        
        if not models:
            raise RoutingError("No suitable models available after filtering")
        
        # Step 3: Apply routing strategy
        decision_start = time.time()
        
        # Determine strategy to use
        strategy_name = routing_request.strategy or settings.default_routing_strategy
        strategy = strategy_factory.get_strategy(strategy_name)
        
        # Add constraints to context
        context = routing_request.context or {}
        if routing_request.max_latency_ms:
            context["max_latency_ms"] = routing_request.max_latency_ms
        if routing_request.max_cost_usd:
            context["max_cost_usd"] = routing_request.max_cost_usd
        
        # Make decision
        decision = await strategy.make_decision(
            prompt=routing_request.prompt,
            analysis=analysis,
            available_models=models,
            context=context
        )
        
        decision_time = (time.time() - decision_start) * 1000
        
        # Step 4: Build complete routing decision
        prompt_hash = hashlib.md5(routing_request.prompt.encode()).hexdigest()
        
        # Estimate token count if not available
        if "prompt_tokens" not in analysis:
            from utils.token_counter import estimate_tokens
            prompt_tokens = estimate_tokens(routing_request.prompt)
        else:
            prompt_tokens = analysis.get("prompt_tokens")
        
        routing_decision = RoutingDecision(
            request_id=request_id,
            timestamp=datetime.now(),
            prompt_hash=prompt_hash,
            prompt_length=len(routing_request.prompt),
            prompt_tokens=prompt_tokens,
            prompt_type=analysis.get("prompt_type", "unknown"),
            complexity_score=analysis.get("complexity_score", 0.5),
            sentiment_score=analysis.get("sentiment_score"),
            language=analysis.get("language", "en"),
            entities=analysis.get("entities"),
            keywords=analysis.get("keywords", []),
            selected_model=decision["selected_model"],
            fallback_model=decision.get("fallback_model"),
            routing_strategy=strategy_name,
            confidence_score=decision.get("confidence", 0.8),
            reasoning=decision.get("reasoning", "No reasoning provided"),
            analysis_time_ms=round(analysis_time, 2),
            decision_time_ms=round(decision_time, 2),
            total_time_ms=round((time.time() - start_time) * 1000, 2),
            alternatives=decision.get("alternatives", []),
            cache_hit=False
        )
        
        # Step 5: Cache the decision (if enabled)
        if settings.cache.enabled:
            background_tasks.add_task(
                _cache_routing_decision,
                cache_key=cache_key,
                decision=routing_decision.dict(),
                ttl=settings.cache.default_ttl
            )
        
        # Step 6: Record metrics
        background_tasks.add_task(
            _record_routing_metrics,
            decision=routing_decision
        )
        
        # Step 7: Log the decision
        background_tasks.add_task(
            routing_repository.log_decision,
            decision=routing_decision.dict()
        )
        
        logger.info(
            "routing_decision_completed",
            request_id=request_id,
            selected_model=routing_decision.selected_model,
            strategy=routing_decision.routing_strategy,
            confidence=routing_decision.confidence_score,
            total_time_ms=routing_decision.total_time_ms
        )
        
        return routing_decision
        
    except RoutingError as e:
        logger.error(
            "routing_decision_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=503,
            detail={
                "error": "RoutingError",
                "message": str(e),
                "request_id": request_id
            }
        )
        
    except Exception as e:
        logger.error(
            "routing_decision_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to make routing decision",
                "request_id": request_id
            }
        )


@router.post(
    "/batch",
    summary="Batch Routing Decisions",
    description="""
    Make routing decisions for multiple prompts in batch.
    
    Processes up to 100 prompts in a single request:
    - Parallel processing for faster results
    - Individual decisions for each prompt
    - Summary statistics
    - Cache utilization
    
    Rate limited: 10 requests per minute
    """,
    response_model=BatchRoutingResponse,
    dependencies=[Depends(rate_limit(limit=10, period=60))]
)
async def batch_routing_decisions(
    request: Request,
    batch_request: BatchRoutingRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> BatchRoutingResponse:
    """
    Make routing decisions for multiple prompts in batch.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "batch_routing_requested",
        request_id=request_id,
        prompt_count=len(batch_request.prompts),
        parallel=batch_request.parallel,
        strategy=batch_request.strategy
    )
    
    try:
        decisions = []
        
        if batch_request.parallel:
            # Process in parallel for speed
            tasks = []
            for prompt in batch_request.prompts:
                routing_request = RoutingRequest(
                    prompt=prompt,
                    strategy=batch_request.strategy
                )
                tasks.append(
                    _make_decision_internal(routing_request, request_id)
                )
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    # Log error but continue processing
                    logger.error(
                        "batch_routing_item_failed",
                        request_id=request_id,
                        error=str(result)
                    )
                else:
                    decisions.append(result)
        else:
            # Process sequentially (less memory pressure)
            for prompt in batch_request.prompts:
                try:
                    routing_request = RoutingRequest(
                        prompt=prompt,
                        strategy=batch_request.strategy
                    )
                    decision = await _make_decision_internal(routing_request, request_id)
                    decisions.append(decision)
                except Exception as e:
                    logger.error(
                        "batch_routing_item_failed",
                        request_id=request_id,
                        error=str(e)
                    )
        
        # Calculate summary statistics
        model_counts = {}
        strategy_counts = {}
        prompt_type_counts = {}
        total_confidence = 0
        total_latency = 0
        
        for decision in decisions:
            model_counts[decision.selected_model] = model_counts.get(decision.selected_model, 0) + 1
            strategy_counts[decision.routing_strategy] = strategy_counts.get(decision.routing_strategy, 0) + 1
            prompt_type_counts[decision.prompt_type] = prompt_type_counts.get(decision.prompt_type, 0) + 1
            total_confidence += decision.confidence_score
            total_latency += decision.total_time_ms
        
        summary = {
            "total_processed": len(batch_request.prompts),
            "successful": len(decisions),
            "failed": len(batch_request.prompts) - len(decisions),
            "model_distribution": model_counts,
            "strategy_distribution": strategy_counts,
            "prompt_type_distribution": prompt_type_counts,
            "average_confidence": total_confidence / len(decisions) if decisions else 0,
            "average_latency_ms": total_latency / len(decisions) if decisions else 0
        }
        
        response = BatchRoutingResponse(
            request_id=request_id,
            timestamp=datetime.now(),
            decisions=decisions,
            summary=summary,
            processing_time_ms=round((time.time() - start_time) * 1000, 2)
        )
        
        logger.info(
            "batch_routing_completed",
            request_id=request_id,
            successful=len(decisions),
            failed=len(batch_request.prompts) - len(decisions),
            processing_time_ms=response.processing_time_ms
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "batch_routing_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to process batch routing request",
                "request_id": request_id
            }
        )


async def _make_decision_internal(
    routing_request: RoutingRequest,
    parent_request_id: str
) -> RoutingDecision:
    """
    Internal helper for making routing decisions in batch.
    """
    # Create a mock request object with a new request ID
    class MockRequest:
        def __init__(self):
            self.state = type('obj', (object,), {
                'request_id': str(uuid.uuid4())
            })
    
    mock_request = MockRequest()
    
    # Call the main decision endpoint
    response = await make_routing_decision(
        request=mock_request,
        routing_request=routing_request,
        background_tasks=BackgroundTasks(),
        api_key="batch-internal"  # This will be overridden in production
    )
    
    return response


# ============================================================================
# ROUTING STRATEGY MANAGEMENT
# ============================================================================

@router.get(
    "/strategies",
    summary="List Routing Strategies",
    description="""
    List all available routing strategies.
    
    Returns detailed information about each strategy:
    - Name and description
    - Configuration weights
    - Recommended use cases
    - Known limitations
    
    Rate limited: 60 requests per minute
    """,
    response_model=List[RoutingStrategyInfo],
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def list_routing_strategies(
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> List[RoutingStrategyInfo]:
    """
    List all available routing strategies.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    strategies = [
        RoutingStrategyInfo(
            name="latency",
            description="Optimize for lowest inference latency",
            weights={"latency": 1.0},
            use_cases=["Real-time applications", "Interactive chat", "User-facing features"],
            limitations=["May use more expensive models", "Doesn't consider cost"]
        ),
        RoutingStrategyInfo(
            name="cost",
            description="Optimize for lowest cost per request",
            weights={"cost": 1.0},
            use_cases=["Batch processing", "Background jobs", "High-volume applications"],
            limitations=["May have higher latency", "May use lower quality models"]
        ),
        RoutingStrategyInfo(
            name="quality",
            description="Optimize for highest response quality",
            weights={"quality": 1.0},
            use_cases=["Creative writing", "Complex reasoning", "Critical applications"],
            limitations=["Higher cost", "Higher latency"]
        ),
        RoutingStrategyInfo(
            name="hybrid",
            description="Balanced optimization across latency, cost, and quality",
            weights=settings.routing_weights,
            use_cases=["General purpose", "Mixed workloads", "Default strategy"],
            limitations=["Not optimal for specific constraints"]
        ),
        RoutingStrategyInfo(
            name="auto",
            description="Automatically select best strategy based on prompt analysis",
            weights=None,
            use_cases=["When unsure which strategy to use", "Dynamic workloads"],
            limitations=["Less predictable", "May change between requests"]
        ),
        RoutingStrategyInfo(
            name="round_robin",
            description="Distribute requests evenly across available models",
            weights=None,
            use_cases=["Load testing", "A/B testing", "Fair resource allocation"],
            limitations=["No optimization", "May use unsuitable models"]
        ),
        RoutingStrategyInfo(
            name="least_connections",
            description="Route to model with fewest active requests",
            weights=None,
            use_cases=["High concurrency", "Variable load patterns"],
            limitations=["No consideration of latency or cost"]
        )
    ]
    
    return strategies


@router.get(
    "/strategies/{strategy_name}",
    summary="Get Strategy Details",
    description="""
    Get detailed information about a specific routing strategy.
    """,
    response_model=RoutingStrategyInfo,
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def get_strategy_details(
    request: Request,
    strategy_name: str,
    api_key: str = Depends(verify_api_key)
) -> RoutingStrategyInfo:
    """
    Get detailed information about a specific routing strategy.
    """
    strategies = await list_routing_strategies(request, api_key)
    
    for strategy in strategies:
        if strategy.name == strategy_name:
            return strategy
    
    raise HTTPException(
        status_code=404,
        detail={
            "error": "StrategyNotFound",
            "message": f"Routing strategy '{strategy_name}' not found"
        }
    )


# ============================================================================
# ROUTING HISTORY AND ANALYTICS
# ============================================================================

@router.get(
    "/decisions",
    summary="Get Routing Decisions",
    description="""
    Get historical routing decisions.
    
    Supports filtering by:
    - Time range
    - Model selected
    - Routing strategy
    - Prompt type
    - Confidence score
    
    Pagination with cursor support for large result sets.
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def get_routing_decisions(
    request: Request,
    start_time: Optional[datetime] = Query(None, description="Start time for filtering"),
    end_time: Optional[datetime] = Query(None, description="End time for filtering"),
    model_id: Optional[str] = Query(None, description="Filter by selected model"),
    strategy: Optional[str] = Query(None, description="Filter by routing strategy"),
    prompt_type: Optional[str] = Query(None, description="Filter by prompt type"),
    min_confidence: Optional[float] = Query(None, description="Minimum confidence score", ge=0, le=1),
    limit: int = Query(100, description="Number of decisions to return", ge=1, le=1000),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Get historical routing decisions with filtering and pagination.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        # Set default time range
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        # Get decisions from repository
        result = await routing_repository.get_decisions(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            strategy=strategy,
            prompt_type=prompt_type,
            min_confidence=min_confidence,
            limit=limit,
            cursor=cursor
        )
        
        return {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "total": result["total"],
            "decisions": result["decisions"],
            "next_cursor": result.get("next_cursor"),
            "query": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "model_id": model_id,
                "strategy": strategy,
                "prompt_type": prompt_type,
                "min_confidence": min_confidence,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(
            "get_routing_decisions_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve routing decisions",
                "request_id": request_id
            }
        )


@router.get(
    "/decisions/{decision_id}",
    summary="Get Routing Decision",
    description="""
    Get a specific routing decision by ID.
    """,
    response_model=RoutingDecision,
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def get_routing_decision(
    request: Request,
    decision_id: str,
    api_key: str = Depends(verify_api_key)
) -> RoutingDecision:
    """
    Get a specific routing decision by ID.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        decision = await routing_repository.get_decision(decision_id)
        
        if not decision:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "DecisionNotFound",
                    "message": f"Routing decision '{decision_id}' not found",
                    "request_id": request_id
                }
            )
        
        return RoutingDecision(**decision)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "get_routing_decision_failed",
            request_id=request_id,
            decision_id=decision_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to retrieve routing decision '{decision_id}'",
                "request_id": request_id
            }
        )


@router.get(
    "/metrics",
    summary="Get Routing Metrics",
    description="""
    Get routing performance metrics.
    
    Returns:
    - Decision volume over time
    - Model selection distribution
    - Strategy effectiveness
    - Confidence trends
    - Cache performance
    - Latency analysis
    
    Supports time range filtering.
    """,
    response_model=RoutingMetricsResponse,
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def get_routing_metrics(
    request: Request,
    period: str = Query("24h", description="Time period (1h, 6h, 24h, 7d, 30d)"),
    api_key: str = Depends(verify_api_key)
) -> RoutingMetricsResponse:
    """
    Get routing performance metrics.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        # Calculate time range
        end_time = datetime.now()
        
        if period == "1h":
            start_time = end_time - timedelta(hours=1)
        elif period == "6h":
            start_time = end_time - timedelta(hours=6)
        elif period == "24h":
            start_time = end_time - timedelta(days=1)
        elif period == "7d":
            start_time = end_time - timedelta(days=7)
        elif period == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(hours=24)
        
        # Get metrics from repository
        metrics = await routing_repository.get_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        # Get hourly trends
        hourly_decisions = await routing_repository.get_hourly_decisions(
            start_time=start_time,
            end_time=end_time
        )
        
        hourly_latency = await routing_repository.get_hourly_latency(
            start_time=start_time,
            end_time=end_time
        )
        
        response = RoutingMetricsResponse(
            timestamp=datetime.now(),
            total_decisions=metrics.get("total_decisions", 0),
            average_confidence=metrics.get("average_confidence", 0),
            average_analysis_time_ms=metrics.get("average_analysis_time_ms", 0),
            average_decision_time_ms=metrics.get("average_decision_time_ms", 0),
            cache_hit_rate=metrics.get("cache_hit_rate", 0),
            model_distribution=metrics.get("model_distribution", {}),
            strategy_distribution=metrics.get("strategy_distribution", {}),
            prompt_type_distribution=metrics.get("prompt_type_distribution", {}),
            strategy_performance=metrics.get("strategy_performance", {}),
            hourly_decisions=hourly_decisions,
            hourly_latency=hourly_latency
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "get_routing_metrics_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve routing metrics",
                "request_id": request_id
            }
        )


# ============================================================================
# CUSTOM ROUTING RULES (ADMIN ONLY)
# ============================================================================

@router.get(
    "/rules",
    summary="List Routing Rules",
    description="""
    List all custom routing rules.
    
    Admin only endpoint.
    """,
    response_model=List[RoutingRule],
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=30, period=60))]
)
async def list_routing_rules(
    request: Request,
    api_key: str = Depends(verify_admin)
) -> List[RoutingRule]:
    """
    List all custom routing rules (admin only).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        rules = await routing_repository.get_all_rules()
        return [RoutingRule(**rule) for rule in rules]
        
    except Exception as e:
        logger.error(
            "list_routing_rules_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to list routing rules",
                "request_id": request_id
            }
        )


@router.post(
    "/rules",
    summary="Create Routing Rule",
    description="""
    Create a custom routing rule.
    
    Rules define conditions that override normal routing logic.
    Example: "If prompt contains 'code', use grok-beta"
    
    Admin only endpoint.
    """,
    response_model=RoutingRule,
    status_code=201,
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=20, period=60))]
)
async def create_routing_rule(
    request: Request,
    rule: RoutingRule,
    api_key: str = Depends(verify_admin)
) -> RoutingRule:
    """
    Create a custom routing rule (admin only).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "create_routing_rule",
        request_id=request_id,
        rule_name=rule.name,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        # Validate rule conditions
        if not rule.conditions:
            raise ValueError("Rule must have at least one condition")
        
        if not rule.action or "model" not in rule.action:
            raise ValueError("Rule must specify a target model")
        
        # Check if model exists
        model_id = rule.action["model"]
        model = await model_registry.get_model(model_id)
        
        if not model:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ModelNotFound",
                    "message": f"Model '{model_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Create rule
        created_rule = await routing_repository.create_rule(
            rule=rule.dict(exclude_none=True)
        )
        
        # Reload router configuration
        await prompt_router.reload_rules()
        
        return RoutingRule(**created_rule)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "InvalidRule",
                "message": str(e),
                "request_id": request_id
            }
        )
    except Exception as e:
        logger.error(
            "create_routing_rule_failed",
            request_id=request_id,
            rule_name=rule.name,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to create routing rule",
                "request_id": request_id
            }
        )


@router.put(
    "/rules/{rule_id}",
    summary="Update Routing Rule",
    description="""
    Update an existing custom routing rule.
    
    Admin only endpoint.
    """,
    response_model=RoutingRule,
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=20, period=60))]
)
async def update_routing_rule(
    request: Request,
    rule_id: str,
    rule: RoutingRule,
    api_key: str = Depends(verify_admin)
) -> RoutingRule:
    """
    Update an existing custom routing rule (admin only).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "update_routing_rule",
        request_id=request_id,
        rule_id=rule_id,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        # Check if rule exists
        existing = await routing_repository.get_rule(rule_id)
        if not existing:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RuleNotFound",
                    "message": f"Routing rule '{rule_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Update rule
        updated_rule = await routing_repository.update_rule(
            rule_id=rule_id,
            updates=rule.dict(exclude_none=True)
        )
        
        # Reload router configuration
        await prompt_router.reload_rules()
        
        return RoutingRule(**updated_rule)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "update_routing_rule_failed",
            request_id=request_id,
            rule_id=rule_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to update routing rule '{rule_id}'",
                "request_id": request_id
            }
        )


@router.delete(
    "/rules/{rule_id}",
    summary="Delete Routing Rule",
    description="""
    Delete a custom routing rule.
    
    Admin only endpoint.
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=20, period=60))]
)
async def delete_routing_rule(
    request: Request,
    rule_id: str,
    api_key: str = Depends(verify_admin)
) -> Dict[str, Any]:
    """
    Delete a custom routing rule (admin only).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "delete_routing_rule",
        request_id=request_id,
        rule_id=rule_id,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        # Check if rule exists
        existing = await routing_repository.get_rule(rule_id)
        if not existing:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RuleNotFound",
                    "message": f"Routing rule '{rule_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Delete rule
        await routing_repository.delete_rule(rule_id)
        
        # Reload router configuration
        await prompt_router.reload_rules()
        
        return {
            "status": "deleted",
            "rule_id": rule_id,
            "message": f"Routing rule '{rule_id}' deleted successfully",
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "delete_routing_rule_failed",
            request_id=request_id,
            rule_id=rule_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to delete routing rule '{rule_id}'",
                "request_id": request_id
            }
        )


@router.post(
    "/rules/reload",
    summary="Reload Routing Rules",
    description="""
    Reload routing rules from configuration.
    
    Admin only endpoint.
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(verify_admin), Depends(rate_limit(limit=10, period=60))]
)
async def reload_routing_rules(
    request: Request,
    api_key: str = Depends(verify_admin)
) -> Dict[str, Any]:
    """
    Reload routing rules from configuration (admin only).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "reload_routing_rules",
        request_id=request_id,
        admin_id=getattr(request.state, "user_id", None)
    )
    
    try:
        # Reload from database
        await prompt_router.reload_rules()
        
        # Also reload from config file
        rules = await prompt_router.load_rules_from_config()
        
        return {
            "status": "reloaded",
            "rules_loaded": len(rules),
            "message": "Routing rules reloaded successfully",
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(
            "reload_routing_rules_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to reload routing rules",
                "request_id": request_id
            }
        )


# ============================================================================
# FEEDBACK LOOP - IMPROVE ROUTING WITH USER FEEDBACK
# ============================================================================

@router.post(
    "/feedback",
    summary="Submit Routing Feedback",
    description="""
    Submit feedback on a routing decision.
    
    User feedback helps improve future routing decisions:
    - Was the selected model appropriate?
    - Was the response quality good?
    - Would another model have been better?
    
    Feedback is used to adjust routing weights and strategies.
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(rate_limit(limit=50, period=60))]
)
async def submit_routing_feedback(
    request: Request,
    decision_id: str = Query(..., description="ID of the routing decision"),
    rating: int = Query(..., description="Rating (1-5)", ge=1, le=5),
    appropriate_model: bool = Query(..., description="Was the selected model appropriate?"),
    better_model: Optional[str] = Query(None, description="Model that would have been better"),
    comments: Optional[str] = Query(None, description="Additional comments"),
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Submit feedback on a routing decision.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "routing_feedback_submitted",
        request_id=request_id,
        decision_id=decision_id,
        rating=rating,
        appropriate_model=appropriate_model
    )
    
    try:
        # Get the original decision
        decision = await routing_repository.get_decision(decision_id)
        
        if not decision:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "DecisionNotFound",
                    "message": f"Routing decision '{decision_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Store feedback
        feedback = {
            "decision_id": decision_id,
            "user_id": getattr(request.state, "user_id", None),
            "rating": rating,
            "appropriate_model": appropriate_model,
            "better_model": better_model,
            "comments": comments,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        await routing_repository.store_feedback(feedback)
        
        # Update routing strategy weights based on feedback
        if settings.enable_adaptive_routing:
            background_tasks.add_task(
                _update_routing_weights_from_feedback,
                feedback=feedback,
                decision=decision
            )
        
        return {
            "status": "success",
            "message": "Feedback received. Thank you for helping improve our routing!",
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "routing_feedback_failed",
            request_id=request_id,
            decision_id=decision_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to submit routing feedback",
                "request_id": request_id
            }
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _generate_cache_key(routing_request: RoutingRequest) -> str:
    """
    Generate a cache key for routing decisions.
    """
    # Create a stable representation of the request
    key_data = {
        "prompt_hash": hashlib.md5(routing_request.prompt.encode()).hexdigest(),
        "strategy": routing_request.strategy,
        "preferred_models": sorted(routing_request.preferred_models) if routing_request.preferred_models else None,
        "excluded_models": sorted(routing_request.excluded_models) if routing_request.excluded_models else None,
        "required_capabilities": sorted(routing_request.required_capabilities) if routing_request.required_capabilities else None,
        "max_latency_ms": routing_request.max_latency_ms,
        "max_cost_usd": routing_request.max_cost_usd
    }
    
    key_json = json.dumps(key_data, sort_keys=True)
    key_hash = hashlib.sha256(key_json.encode()).hexdigest()
    
    return f"routing:decision:{key_hash}"


async def _cache_routing_decision(
    cache_key: str,
    decision: dict,
    ttl: int
):
    """
    Cache a routing decision.
    """
    await cache_manager.set(cache_key, decision, ttl=ttl)


async def _record_routing_metrics(decision: RoutingDecision):
    """
    Record routing decision metrics.
    """
    await metrics_collector.record_routing_decision(
        decision=decision.dict()
    )


async def _update_routing_weights_from_feedback(
    feedback: dict,
    decision: dict
):
    """
    Update routing strategy weights based on user feedback.
    Adaptive routing - improves over time.
    """
    # TODO: Implement adaptive weight adjustment
    # This would use reinforcement learning or simple heuristics
    # to adjust strategy weights based on feedback ratings
    pass


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "router",
    "RoutingRequest",
    "RoutingDecision",
    "BatchRoutingRequest",
    "BatchRoutingResponse",
    "RoutingStrategyInfo",
    "RoutingMetricsResponse",
    "RoutingRule"
]