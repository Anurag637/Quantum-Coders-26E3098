"""
Routing Service - Production Ready
Orchestrates intelligent prompt routing with multiple strategies,
real-time model scoring, fallback chains, and adaptive optimization.
"""

import asyncio
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from core.logging import get_logger
from core.exceptions import RoutingError, NoSuitableModelError
from core.circuit_breaker import CircuitBreakerRegistry, CircuitBreakerConfig
from config import settings
from models.model_manager import ModelManager
from models.model_registry import ModelRegistry
from router.prompt_analyzer import PromptAnalyzer
from router.routing_strategies import (
    RoutingStrategyFactory,
    LatencyOptimizedStrategy,
    CostOptimizedStrategy,
    QualityOptimizedStrategy,
    HybridStrategy,
    AdaptiveStrategy
)
from cache.cache_manager import CacheManager
from cache.semantic_cache import SemanticCache
from monitoring.metrics import MetricsCollector
from database.repositories.routing_repository import RoutingRepository
from database.repositories.model_repository import ModelRepository
from services.cost_calculator import CostCalculator
from services.token_counter import TokenCounter

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# ROUTING SERVICE
# ============================================================================

class RoutingService:
    """
    Intelligent routing service for LLM inference.
    
    Features:
    - Multiple routing strategies with real-time scoring
    - Prompt analysis and classification
    - Model capability matching
    - Fallback chains with automatic failover
    - Decision caching for performance
    - Adaptive weight optimization
    - Comprehensive metrics and monitoring
    - Circuit breaker protection
    """
    
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        model_registry: Optional[ModelRegistry] = None,
        prompt_analyzer: Optional[PromptAnalyzer] = None,
        cache_manager: Optional[CacheManager] = None,
        semantic_cache: Optional[SemanticCache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        routing_repository: Optional[RoutingRepository] = None,
        model_repository: Optional[ModelRepository] = None
    ):
        # Initialize dependencies
        self.model_manager = model_manager or ModelManager()
        self.model_registry = model_registry or ModelRegistry()
        self.prompt_analyzer = prompt_analyzer or PromptAnalyzer()
        self.cache_manager = cache_manager or CacheManager()
        self.semantic_cache = semantic_cache or SemanticCache()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.routing_repository = routing_repository or RoutingRepository()
        self.model_repository = model_repository or ModelRepository()
        
        # Initialize utilities
        self.cost_calculator = CostCalculator()
        self.token_counter = TokenCounter()
        
        # Initialize strategy factory
        self.strategy_factory = RoutingStrategyFactory()
        
        # Initialize circuit breakers
        self.circuit_breaker_registry = CircuitBreakerRegistry()
        
        # Routing configuration
        self.config = {
            "default_strategy": settings.default_routing_strategy.value,
            "weights": settings.routing_weights,
            "min_confidence": 0.6,
            "cache_enabled": True,
            "cache_ttl": 3600,
            "max_candidates": 10,
            "fallback_enabled": True,
            "default_fallback": "mistral-7b-instruct",
            "adaptive_enabled": True,
            "adaptation_rate": 0.1
        }
        
        # Strategy instances cache
        self.strategies = {}
        
        # Model performance history (for adaptive routing)
        self.model_performance: Dict[str, Dict[str, Any]] = {}
        
        # Routing statistics
        self.stats = {
            "total_decisions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fallback_used": 0,
            "avg_confidence": 0.0,
            "avg_latency_ms": 0.0
        }
        
        logger.info(
            "routing_service_initialized",
            default_strategy=self.config["default_strategy"],
            strategies=self.strategy_factory.list_strategies(),
            cache_enabled=self.config["cache_enabled"],
            adaptive_enabled=self.config["adaptive_enabled"]
        )
    
    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================
    
    async def route_prompt(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        strategy: Optional[str] = None,
        user: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        skip_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Route a prompt to the optimal model.
        
        Args:
            prompt: Input prompt
            messages: Full conversation context
            strategy: Routing strategy to use
            user: User information
            context: Additional context
            skip_cache: Skip cache lookup
        
        Returns:
            Routing decision with selected model and metadata
        
        Raises:
            RoutingError: If routing fails
            NoSuitableModelError: If no suitable model found
        """
        start_time = time.time()
        self.stats["total_decisions"] += 1
        
        request_id = hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:16]
        
        logger.debug(
            "routing_request_started",
            request_id=request_id,
            prompt_length=len(prompt),
            strategy=strategy or self.config["default_strategy"]
        )
        
        try:
            # ====================================================================
            # STEP 1: Check cache (if enabled)
            # ====================================================================
            if self.config["cache_enabled"] and not skip_cache:
                cached_decision = await self._get_cached_decision(
                    prompt=prompt,
                    strategy=strategy,
                    user=user
                )
                
                if cached_decision:
                    self.stats["cache_hits"] += 1
                    
                    logger.debug(
                        "routing_cache_hit",
                        request_id=request_id,
                        model=cached_decision["selected_model"],
                        confidence=cached_decision["confidence"]
                    )
                    
                    # Update cache hit metric
                    await self.metrics_collector.record_routing_cache_hit(
                        strategy=cached_decision.get("strategy", "unknown")
                    )
                    
                    return cached_decision
            
            self.stats["cache_misses"] += 1
            
            # ====================================================================
            # STEP 2: Analyze prompt
            # ====================================================================
            analysis = await self.prompt_analyzer.analyze(prompt)
            
            # Add conversation context if available
            if messages:
                analysis["conversation_length"] = len(messages)
                analysis["has_system_message"] = any(
                    msg.get("role") == "system" for msg in messages
                )
            
            # ====================================================================
            # STEP 3: Get available models
            # ====================================================================
            available_models = await self._get_available_models(
                user=user,
                analysis=analysis,
                context=context
            )
            
            if not available_models:
                raise NoSuitableModelError(
                    criteria={"status": "ready"},
                    detail="No models available for inference"
                )
            
            # ====================================================================
            # STEP 4: Select routing strategy
            # ====================================================================
            strategy_name = strategy or self.config["default_strategy"]
            routing_strategy = await self._get_strategy(strategy_name)
            
            # ====================================================================
            # STEP 5: Score and rank models
            # ====================================================================
            scored_models = await self._score_models(
                models=available_models,
                analysis=analysis,
                strategy=routing_strategy,
                user=user,
                context=context
            )
            
            if not scored_models:
                # Try fallback strategy
                if self.config["fallback_enabled"]:
                    logger.warning(
                        "no_models_for_strategy",
                        strategy=strategy_name,
                        fallback="hybrid"
                    )
                    
                    routing_strategy = await self._get_strategy("hybrid")
                    scored_models = await self._score_models(
                        models=available_models,
                        analysis=analysis,
                        strategy=routing_strategy,
                        user=user,
                        context=context
                    )
            
            if not scored_models:
                raise NoSuitableModelError(
                    criteria={"strategy": strategy_name},
                    detail=f"No models meet criteria for strategy: {strategy_name}"
                )
            
            # ====================================================================
            # STEP 6: Select primary model
            # ====================================================================
            primary_model = scored_models[0]
            
            # Check circuit breaker for selected model
            if await self._is_circuit_open(primary_model["model_id"]):
                logger.warning(
                    "circuit_open",
                    model=primary_model["model_id"],
                    fallback="enabled"
                )
                
                # Try fallback models
                primary_model = await self._get_fallback_model(
                    primary_model=primary_model,
                    scored_models=scored_models[1:],
                    analysis=analysis
                )
                
                if primary_model:
                    self.stats["fallback_used"] += 1
                else:
                    raise NoSuitableModelError(
                        criteria={"circuit_status": "open"},
                        detail="Selected model circuit is open and no fallbacks available"
                    )
            
            # ====================================================================
            # STEP 7: Prepare routing decision
            # ====================================================================
            decision = await self._create_decision(
                request_id=request_id,
                prompt=prompt,
                analysis=analysis,
                selected_model=primary_model,
                strategy=strategy_name,
                alternatives=scored_models[1:4],  # Top 3 alternatives
                latency_ms=(time.time() - start_time) * 1000,
                cache_hit=False
            )
            
            # ====================================================================
            # STEP 8: Cache decision
            # ====================================================================
            if self.config["cache_enabled"]:
                await self._cache_decision(
                    prompt=prompt,
                    strategy=strategy_name,
                    user=user,
                    decision=decision
                )
            
            # ====================================================================
            # STEP 9: Update statistics and metrics
            # ====================================================================
            await self._update_statistics(decision)
            await self._record_metrics(decision)
            
            # ====================================================================
            # STEP 10: Log routing decision
            # ====================================================================
            await self.routing_repository.log_decision({
                "request_id": request_id,
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
                "prompt_length": len(prompt),
                "prompt_type": analysis.get("prompt_type", "unknown"),
                "complexity_score": analysis.get("complexity_score", 0.5),
                "selected_model": decision["selected_model"],
                "routing_strategy": decision["strategy"],
                "confidence_score": decision["confidence"],
                "reasoning": decision["reasoning"],
                "analysis_time_ms": analysis.get("analysis_time_ms", 0),
                "decision_time_ms": decision["latency_ms"],
                "total_time_ms": decision["latency_ms"],
                "cache_hit": False
            })
            
            logger.info(
                "routing_decision_made",
                request_id=request_id,
                model=decision["selected_model"],
                strategy=decision["strategy"],
                confidence=decision["confidence"],
                latency_ms=round(decision["latency_ms"], 2)
            )
            
            return decision
            
        except NoSuitableModelError:
            raise
        except Exception as e:
            logger.error(
                "routing_failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            raise RoutingError(
                detail=f"Routing failed: {str(e)}",
                request_id=request_id
            )
    
    async def route_batch(
        self,
        prompts: List[str],
        strategy: Optional[str] = None,
        user: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        max_concurrency: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Route multiple prompts in parallel.
        
        Args:
            prompts: List of prompts to route
            strategy: Routing strategy to use
            user: User information
            context: Additional context
            max_concurrency: Maximum concurrent routing operations
        
        Returns:
            List of routing decisions
        """
        logger.info(
            "batch_routing_started",
            prompt_count=len(prompts),
            strategy=strategy,
            max_concurrency=max_concurrency
        )
        
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def route_with_semaphore(prompt: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.route_prompt(
                        prompt=prompt,
                        strategy=strategy,
                        user=user,
                        context=context
                    )
                except Exception as e:
                    logger.error(f"Batch routing failed for prompt: {e}")
                    return {
                        "error": str(e),
                        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt
                    }
        
        # Route all prompts concurrently
        tasks = [route_with_semaphore(prompt) for prompt in prompts]
        decisions = await asyncio.gather(*tasks)
        
        logger.info(
            "batch_routing_completed",
            successful=sum(1 for d in decisions if "error" not in d),
            failed=sum(1 for d in decisions if "error" in d)
        )
        
        return decisions
    
    # ========================================================================
    # MODEL SCORING AND SELECTION
    # ========================================================================
    
    async def _get_available_models(
        self,
        user: Optional[Dict[str, Any]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get list of available models for routing."""
        # Get all models from registry
        all_models, _ = await self.model_repository.get_all_models(
            status="ready",
            is_loaded=True
        )
        
        # Also include external API models
        external_models, _ = await self.model_repository.get_all_models(
            model_type="external",
            status="available"
        )
        
        all_models.extend(external_models)
        
        available_models = []
        
        for model in all_models:
            # Check user tier access
            if not self._check_user_access(model, user):
                continue
            
            # Check required capabilities
            if analysis and not self._check_capabilities(model, analysis):
                continue
            
            # Get real-time metrics
            model_id = model["model_id"]
            
            # Get performance history for adaptive routing
            if self.config["adaptive_enabled"]:
                performance = self.model_performance.get(model_id, {})
                model["performance"] = performance
            
            # Check circuit breaker status
            if await self._is_circuit_open(model_id):
                model["circuit_open"] = True
                model["status"] = "circuit_open"
            
            available_models.append(model)
        
        return available_models
    
    def _check_user_access(
        self,
        model: Dict[str, Any],
        user: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if user has access to model."""
        # Admin can access everything
        if user and user.get("is_admin"):
            return True
        
        # Check model access tier
        access_tier = model.get("metadata", {}).get("access_tier", "free")
        
        if not user:
            return access_tier == "free"
        
        user_tier = user.get("tier", "free")
        
        tiers = ["free", "pro", "enterprise"]
        user_level = tiers.index(user_tier) if user_tier in tiers else 0
        model_level = tiers.index(access_tier) if access_tier in tiers else 0
        
        return user_level >= model_level
    
    def _check_capabilities(
        self,
        model: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> bool:
        """Check if model has required capabilities."""
        required = analysis.get("required_capabilities", [])
        model_caps = model.get("capabilities", [])
        
        if not required:
            return True
        
        return all(cap in model_caps for cap in required)
    
    async def _score_models(
        self,
        models: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        strategy: Any,
        user: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Score and rank models based on strategy."""
        scored_models = []
        
        for model in models:
            # Skip models with open circuit
            if model.get("circuit_open", False):
                continue
            
            # Calculate base score
            score = await strategy.calculate_score(
                model=model,
                analysis=analysis,
                user=user
            )
            
            # Apply adaptive adjustments if enabled
            if self.config["adaptive_enabled"]:
                score = self._apply_adaptive_adjustment(model, score)
            
            # Skip models below confidence threshold
            if score >= self.config["min_confidence"]:
                scored_models.append({
                    "model_id": model["model_id"],
                    "model_name": model.get("name", model["model_id"]),
                    "provider": model.get("provider", "unknown"),
                    "type": model.get("type", "unknown"),
                    "score": score,
                    "capabilities": model.get("capabilities", []),
                    "latency_ms": model.get("latency_p95_ms", 500),
                    "cost_per_token": model.get("cost_per_token", 0),
                    "quality_score": model.get("quality_score", 0.7),
                    "circuit_open": model.get("circuit_open", False),
                    "metadata": model.get("metadata", {})
                })
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_models[:self.config["max_candidates"]]
    
    def _apply_adaptive_adjustment(
        self,
        model: Dict[str, Any],
        base_score: float
    ) -> float:
        """Apply adaptive weight adjustment based on performance history."""
        model_id = model["model_id"]
        
        if model_id not in self.model_performance:
            return base_score
        
        perf = self.model_performance[model_id]
        adjustment = 0.0
        
        # Boost for good reliability
        if perf.get("success_rate", 1.0) > 0.99:
            adjustment += 0.05
        elif perf.get("success_rate", 1.0) < 0.95:
            adjustment -= 0.1
        
        # Boost for low latency
        avg_latency = perf.get("avg_latency_ms", 500)
        if avg_latency < 200:
            adjustment += 0.05
        elif avg_latency > 1000:
            adjustment -= 0.1
        
        # Boost for cost efficiency
        avg_cost = perf.get("avg_cost_per_token", 0.00001)
        if avg_cost < 0.000001:
            adjustment += 0.05
        elif avg_cost > 0.0001:
            adjustment -= 0.1
        
        return max(0.0, min(1.0, base_score + adjustment))
    
    # ========================================================================
    # FALLBACK HANDLING
    # ========================================================================
    
    async def _get_fallback_model(
        self,
        primary_model: Dict[str, Any],
        scored_models: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get fallback model when primary is unavailable."""
        # Try next best models
        for candidate in scored_models:
            if not await self._is_circuit_open(candidate["model_id"]):
                logger.info(
                    "fallback_model_selected",
                    primary=primary_model["model_id"],
                    fallback=candidate["model_id"],
                    score=candidate["score"]
                )
                return candidate
        
        # Try default fallback model
        default = self.config["default_fallback"]
        
        # Check if default is available
        model = await self.model_repository.get_model(default)
        if model and model.get("status") in ["ready", "available"]:
            if not await self._is_circuit_open(default):
                logger.info(
                    "default_fallback_selected",
                    fallback=default
                )
                return {
                    "model_id": default,
                    "model_name": model.get("name", default),
                    "score": 0.5,
                    "fallback": True
                }
        
        return None
    
    async def _is_circuit_open(self, model_id: str) -> bool:
        """Check if circuit breaker is open for model."""
        circuit_breaker = self.circuit_breaker_registry.get(
            name=f"model:{model_id}"
        )
        return circuit_breaker.is_open
    
    # ========================================================================
    # DECISION CREATION AND CACHING
    # ========================================================================
    
    async def _create_decision(
        self,
        request_id: str,
        prompt: str,
        analysis: Dict[str, Any],
        selected_model: Dict[str, Any],
        strategy: str,
        alternatives: List[Dict[str, Any]],
        latency_ms: float,
        cache_hit: bool = False
    ) -> Dict[str, Any]:
        """Create a routing decision dictionary."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            selected_model=selected_model,
            strategy=strategy,
            analysis=analysis
        )
        
        return {
            "request_id": request_id,
            "prompt_hash": prompt_hash,
            "prompt_length": len(prompt),
            "prompt_type": analysis.get("prompt_type", "unknown"),
            "complexity_score": analysis.get("complexity_score", 0.5),
            "selected_model": selected_model["model_id"],
            "selected_model_name": selected_model.get("model_name", selected_model["model_id"]),
            "strategy": strategy,
            "confidence": selected_model["score"],
            "reasoning": reasoning,
            "alternatives": [
                {
                    "model_id": alt["model_id"],
                    "model_name": alt.get("model_name", alt["model_id"]),
                    "score": alt["score"]
                }
                for alt in alternatives[:3]
            ],
            "latency_ms": round(latency_ms, 2),
            "cache_hit": cache_hit,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    def _generate_reasoning(
        self,
        selected_model: Dict[str, Any],
        strategy: str,
        analysis: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for routing decision."""
        model_name = selected_model.get("model_name", selected_model["model_id"])
        score = selected_model["score"]
        prompt_type = analysis.get("prompt_type", "general")
        
        parts = [
            f"Selected {model_name} for {prompt_type} task",
            f"(confidence: {score:.2f})",
            f"using {strategy} strategy."
        ]
        
        # Add specific reasoning based on strategy
        if strategy == "latency":
            latency = selected_model.get("latency_ms", "unknown")
            parts.append(f"Expected latency: {latency}ms")
        elif strategy == "cost":
            cost = selected_model.get("cost_per_token", 0)
            parts.append(f"Cost per token: ${cost:.6f}")
        elif strategy == "quality":
            quality = selected_model.get("quality_score", 0.7)
            parts.append(f"Quality score: {quality:.2f}")
        
        # Add capability match if applicable
        if prompt_type in selected_model.get("capabilities", []):
            parts.append(f"Specialized in {prompt_type} tasks")
        
        # Add fallback indicator
        if selected_model.get("fallback", False):
            parts.append("(fallback model)")
        
        return " ".join(parts)
    
    async def _get_cached_decision(
        self,
        prompt: str,
        strategy: Optional[str],
        user: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get cached routing decision."""
        cache_key = self._generate_cache_key(prompt, strategy, user)
        return await self.cache_manager.get(cache_key)
    
    async def _cache_decision(
        self,
        prompt: str,
        strategy: Optional[str],
        user: Optional[Dict[str, Any]],
        decision: Dict[str, Any]
    ):
        """Cache routing decision."""
        cache_key = self._generate_cache_key(prompt, strategy, user)
        await self.cache_manager.set(
            key=cache_key,
            value=decision,
            ttl=self.config["cache_ttl"]
        )
    
    def _generate_cache_key(
        self,
        prompt: str,
        strategy: Optional[str],
        user: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for routing decision."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        parts = [
            "routing",
            prompt_hash,
            strategy or self.config["default_strategy"]
        ]
        
        if user:
            parts.append(user.get("tier", "free"))
            if user.get("is_admin"):
                parts.append("admin")
        
        return ":".join(parts)
    
    # ========================================================================
    # STRATEGY MANAGEMENT
    # ========================================================================
    
    async def _get_strategy(self, name: str) -> Any:
        """Get or create routing strategy instance."""
        if name not in self.strategies:
            if name == "hybrid":
                self.strategies[name] = HybridStrategy(
                    weights=self.config["weights"]
                )
            elif name == "adaptive":
                self.strategies[name] = AdaptiveStrategy(
                    adaptation_rate=self.config["adaptation_rate"]
                )
            else:
                self.strategies[name] = self.strategy_factory.get_strategy(name)
        
        return self.strategies[name]
    
    async def get_available_strategies(self) -> List[Dict[str, Any]]:
        """Get list of available routing strategies with descriptions."""
        strategies = []
        
        for name in self.strategy_factory.list_strategies():
            strategy_info = {
                "name": name,
                "description": self._get_strategy_description(name),
                "default": name == self.config["default_strategy"]
            }
            
            if name == "hybrid":
                strategy_info["weights"] = self.config["weights"]
            
            strategies.append(strategy_info)
        
        return strategies
    
    def _get_strategy_description(self, strategy: str) -> str:
        """Get description for a routing strategy."""
        descriptions = {
            "latency": "Optimize for lowest inference latency - best for real-time applications",
            "cost": "Optimize for lowest cost per request - best for batch processing",
            "quality": "Optimize for highest response quality - best for critical tasks",
            "hybrid": "Balanced optimization across latency, cost, and quality",
            "round_robin": "Distribute requests evenly across available models",
            "least_connections": "Route to model with fewest active connections",
            "adaptive": "Dynamically adjust weights based on performance history"
        }
        
        return descriptions.get(strategy, "No description available")
    
    async def update_strategy_weights(self, weights: Dict[str, float]) -> bool:
        """Update hybrid strategy weights."""
        try:
            # Validate weights sum to 1.0
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Weights must sum to 1.0, got {total}")
            
            self.config["weights"] = weights
            
            # Update hybrid strategy instance
            if "hybrid" in self.strategies:
                self.strategies["hybrid"] = HybridStrategy(weights=weights)
            
            logger.info("strategy_weights_updated", weights=weights)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update strategy weights: {e}")
            return False
    
    # ========================================================================
    # PERFORMANCE TRACKING
    # ========================================================================
    
    async def update_model_performance(
        self,
        model_id: str,
        latency_ms: float,
        tokens: int,
        cost: float,
        success: bool
    ):
        """Update performance history for a model."""
        if model_id not in self.model_performance:
            self.model_performance[model_id] = {
                "latency_samples": [],
                "tokens_samples": [],
                "cost_samples": [],
                "success_count": 0,
                "total_count": 0,
                "avg_latency_ms": latency_ms,
                "avg_tokens": tokens,
                "avg_cost_per_token": cost / tokens if tokens > 0 else 0,
                "success_rate": 1.0 if success else 0.0
            }
        
        perf = self.model_performance[model_id]
        
        # Keep last 100 samples
        perf["latency_samples"].append(latency_ms)
        if len(perf["latency_samples"]) > 100:
            perf["latency_samples"].pop(0)
        
        perf["tokens_samples"].append(tokens)
        if len(perf["tokens_samples"]) > 100:
            perf["tokens_samples"].pop(0)
        
        perf["cost_samples"].append(cost)
        if len(perf["cost_samples"]) > 100:
            perf["cost_samples"].pop(0)
        
        perf["total_count"] += 1
        if success:
            perf["success_count"] += 1
        
        # Update averages
        perf["avg_latency_ms"] = sum(perf["latency_samples"]) / len(perf["latency_samples"])
        perf["avg_tokens"] = sum(perf["tokens_samples"]) / len(perf["tokens_samples"])
        
        avg_cost = sum(perf["cost_samples"]) / len(perf["cost_samples"])
        avg_tokens = perf["avg_tokens"]
        perf["avg_cost_per_token"] = avg_cost / avg_tokens if avg_tokens > 0 else 0
        
        perf["success_rate"] = perf["success_count"] / perf["total_count"]
        
        # Update adaptive strategy with performance data
        if self.config["adaptive_enabled"] and "adaptive" in self.strategies:
            adaptive = self.strategies["adaptive"]
            adaptive.update_performance(
                model_id=model_id,
                latency_ms=latency_ms,
                cost=cost,
                quality_score=1.0 if success else 0.5  # Simplified quality score
            )
    
    # ========================================================================
    # STATISTICS AND METRICS
    # ========================================================================
    
    async def _update_statistics(self, decision: Dict[str, Any]):
        """Update routing statistics."""
        self.stats["avg_confidence"] = (
            (self.stats["avg_confidence"] * (self.stats["total_decisions"] - 1) +
             decision["confidence"]) / self.stats["total_decisions"]
        )
        
        self.stats["avg_latency_ms"] = (
            (self.stats["avg_latency_ms"] * (self.stats["total_decisions"] - 1) +
             decision["latency_ms"]) / self.stats["total_decisions"]
        )
    
    async def _record_metrics(self, decision: Dict[str, Any]):
        """Record routing decision metrics."""
        await self.metrics_collector.record_routing_decision(
            decision=decision
        )
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get routing service statistics."""
        hit_rate = (
            self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
            if self.stats["cache_hits"] + self.stats["cache_misses"] > 0
            else 0
        )
        
        return {
            "total_decisions": self.stats["total_decisions"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_hit_rate": round(hit_rate * 100, 2),
            "fallback_used": self.stats["fallback_used"],
            "avg_confidence": round(self.stats["avg_confidence"], 3),
            "avg_latency_ms": round(self.stats["avg_latency_ms"], 2),
            "config": self.config,
            "model_performance_count": len(self.model_performance),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    async def reset_statistics(self):
        """Reset routing statistics."""
        self.stats = {
            "total_decisions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fallback_used": 0,
            "avg_confidence": 0.0,
            "avg_latency_ms": 0.0
        }
        logger.info("routing_statistics_reset")
    
    async def clear_cache(self) -> int:
        """Clear routing decision cache."""
        pattern = "routing:*"
        keys = await self.cache_manager.keys(pattern)
        
        if keys:
            await self.cache_manager.delete(*keys)
            return len(keys)
        
        return 0


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_routing_service = None


def get_routing_service() -> RoutingService:
    """Get singleton routing service instance."""
    global _routing_service
    if not _routing_service:
        _routing_service = RoutingService()
    return _routing_service


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "RoutingService",
    "get_routing_service"
]