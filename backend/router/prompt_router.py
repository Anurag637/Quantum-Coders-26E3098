"""
Prompt Router - Production Ready
Intelligent routing of prompts to optimal models based on analysis,
capabilities, performance, cost, and user preferences.
"""

import time
import uuid
import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from core.logging import get_logger
from core.exceptions import RoutingError, NoSuitableModelError
from config import settings
from models.model_manager import ModelManager
from models.model_registry import ModelRegistry
from router.prompt_analyzer import PromptAnalyzer
from router.routing_strategies import (
    RoutingStrategy,
    LatencyOptimizedStrategy,
    CostOptimizedStrategy,
    QualityOptimizedStrategy,
    HybridStrategy,
    RoundRobinStrategy,
    LeastConnectionsStrategy
)
from monitoring.metrics import MetricsCollector
from cache.cache_manager import CacheManager

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# ROUTING CONFIGURATION
# ============================================================================

class RoutingConfig:
    """Configuration for prompt routing."""
    
    def __init__(self):
        # Default strategy
        self.default_strategy = settings.default_routing_strategy.value
        
        # Strategy weights (for hybrid)
        self.weights = settings.routing_weights
        
        # Minimum confidence score to use routing decision
        self.min_confidence = 0.6
        
        # Cache routing decisions
        self.cache_enabled = True
        self.cache_ttl = 3600  # 1 hour
        
        # Model selection constraints
        self.max_candidates = 10
        self.fallback_enabled = True
        self.default_fallback_model = "mistral-7b-instruct"
        
        # Performance thresholds
        self.max_latency_ms = 2000
        self.max_cost_usd = 0.01
        self.min_quality_score = 0.7


# ============================================================================
# ROUTING DECISION
# ============================================================================

class RoutingDecision:
    """Encapsulates a routing decision with metadata."""
    
    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_hash: str,
        analysis: Dict[str, Any],
        selected_model: str,
        strategy: str,
        confidence: float,
        reasoning: str,
        alternatives: List[Dict[str, Any]],
        latency: float,
        cache_hit: bool = False
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_hash = prompt_hash
        self.analysis = analysis
        self.selected_model = selected_model
        self.strategy = strategy
        self.confidence = confidence
        self.reasoning = reasoning
        self.alternatives = alternatives
        self.latency = latency
        self.cache_hit = cache_hit
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            "request_id": self.request_id,
            "prompt_hash": self.prompt_hash,
            "prompt_length": len(self.prompt),
            "prompt_type": self.analysis.get("prompt_type", "unknown"),
            "complexity_score": self.analysis.get("complexity_score", 0.5),
            "selected_model": self.selected_model,
            "strategy": self.strategy,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives[:3],  # Top 3 alternatives
            "latency_ms": round(self.latency * 1000, 2),
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        return f"RoutingDecision(model={self.selected_model}, confidence={self.confidence:.2f}, strategy={self.strategy})"


# ============================================================================
# PROMPT ROUTER
# ============================================================================

class PromptRouter:
    """
    Intelligent prompt router for LLM inference.
    
    Features:
    - Multiple routing strategies (latency, cost, quality, hybrid)
    - Prompt analysis and classification
    - Model capability matching
    - Fallback chains
    - Decision caching
    - Performance monitoring
    """
    
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        model_registry: Optional[ModelRegistry] = None,
        prompt_analyzer: Optional[PromptAnalyzer] = None,
        cache_manager: Optional[CacheManager] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        # Initialize components
        self.model_manager = model_manager or ModelManager()
        self.model_registry = model_registry or ModelRegistry()
        self.prompt_analyzer = prompt_analyzer or PromptAnalyzer()
        self.cache_manager = cache_manager or CacheManager()
        self.metrics_collector = metrics_collector or MetricsCollector()
        
        # Configuration
        self.config = RoutingConfig()
        
        # Initialize routing strategies
        self.strategies = {
            "latency": LatencyOptimizedStrategy(),
            "cost": CostOptimizedStrategy(),
            "quality": QualityOptimizedStrategy(),
            "hybrid": HybridStrategy(weights=self.config.weights),
            "round_robin": RoundRobinStrategy(),
            "least_connections": LeastConnectionsStrategy()
        }
        
        # Custom routing rules (loaded from config)
        self.custom_rules = []
        self._load_custom_rules()
        
        logger.info(
            "prompt_router_initialized",
            default_strategy=self.config.default_strategy,
            strategies=list(self.strategies.keys()),
            cache_enabled=self.config.cache_enabled
        )
    
    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================
    
    async def make_decision(
        self,
        prompt: str,
        analysis: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None,
        user: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a routing decision for a prompt.
        
        Args:
            prompt: Input prompt
            analysis: Pre-computed prompt analysis (optional)
            strategy: Routing strategy to use
            user: User information (for tier-based routing)
            metadata: Additional metadata
        
        Returns:
            Routing decision dictionary
        
        Raises:
            RoutingError: If routing fails
            NoSuitableModelError: If no suitable model found
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        logger.debug(
            "routing_decision_started",
            request_id=request_id,
            prompt_length=len(prompt),
            strategy=strategy or self.config.default_strategy
        )
        
        try:
            # ====================================================================
            # STEP 1: Generate cache key and check cache
            # ====================================================================
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            cache_key = self._generate_cache_key(
                prompt_hash=prompt_hash,
                strategy=strategy,
                user=user
            )
            
            if self.config.cache_enabled:
                cached_decision = await self._get_cached_decision(cache_key)
                if cached_decision:
                    logger.debug(
                        "routing_cache_hit",
                        request_id=request_id,
                        cache_key=cache_key
                    )
                    return cached_decision
            
            # ====================================================================
            # STEP 2: Analyze prompt if not provided
            # ====================================================================
            if not analysis:
                analysis = await self.prompt_analyzer.analyze(prompt)
            
            # ====================================================================
            # STEP 3: Apply custom routing rules
            # ====================================================================
            custom_decision = await self._apply_custom_rules(
                prompt=prompt,
                analysis=analysis,
                user=user
            )
            
            if custom_decision:
                logger.debug(
                    "custom_rule_applied",
                    request_id=request_id,
                    rule=custom_decision.get("rule"),
                    model=custom_decision.get("selected_model")
                )
                return custom_decision
            
            # ====================================================================
            # STEP 4: Get available models
            # ====================================================================
            available_models = await self._get_available_models(
                user=user,
                analysis=analysis
            )
            
            if not available_models:
                raise NoSuitableModelError(
                    criteria={"status": "ready"},
                    detail="No models available for inference"
                )
            
            # ====================================================================
            # STEP 5: Select routing strategy
            # ====================================================================
            strategy_name = strategy or self.config.default_strategy
            routing_strategy = self.strategies.get(strategy_name)
            
            if not routing_strategy:
                logger.warning(
                    "unknown_strategy",
                    strategy=strategy_name,
                    default=self.config.default_strategy
                )
                routing_strategy = self.strategies[self.config.default_strategy]
                strategy_name = self.config.default_strategy
            
            # ====================================================================
            # STEP 6: Score and rank models
            # ====================================================================
            scored_models = await self._score_models(
                models=available_models,
                analysis=analysis,
                strategy=routing_strategy,
                user=user
            )
            
            if not scored_models:
                raise NoSuitableModelError(
                    criteria={"strategy": strategy_name},
                    detail=f"No models meet criteria for strategy: {strategy_name}"
                )
            
            # ====================================================================
            # STEP 7: Select best model
            # ====================================================================
            selected_model = scored_models[0]
            alternatives = scored_models[1:4]  # Top 3 alternatives
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                selected_model=selected_model,
                strategy=strategy_name,
                analysis=analysis
            )
            
            # ====================================================================
            # STEP 8: Create routing decision
            # ====================================================================
            decision = RoutingDecision(
                request_id=request_id,
                prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                prompt_hash=prompt_hash,
                analysis=analysis,
                selected_model=selected_model["model_id"],
                strategy=strategy_name,
                confidence=selected_model["score"],
                reasoning=reasoning,
                alternatives=alternatives,
                latency=time.time() - start_time,
                cache_hit=False
            )
            
            decision_dict = decision.to_dict()
            
            # ====================================================================
            # STEP 9: Cache decision
            # ====================================================================
            if self.config.cache_enabled:
                await self._cache_decision(cache_key, decision_dict)
            
            # ====================================================================
            # STEP 10: Record metrics
            # ====================================================================
            await self._record_metrics(decision)
            
            logger.info(
                "routing_decision_made",
                request_id=request_id,
                model=selected_model["model_id"],
                strategy=strategy_name,
                confidence=selected_model["score"],
                latency_ms=round(decision.latency * 1000, 2)
            )
            
            return decision_dict
            
        except NoSuitableModelError:
            raise
        except Exception as e:
            logger.error(
                "routing_decision_failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            raise RoutingError(
                detail=f"Failed to make routing decision: {str(e)}"
            )
    
    async def get_strategy(self, name: str) -> Optional[RoutingStrategy]:
        """Get routing strategy by name."""
        return self.strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List available routing strategies."""
        return list(self.strategies.keys())
    
    # ========================================================================
    # MODEL SCORING
    # ========================================================================
    
    async def _score_models(
        self,
        models: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        strategy: RoutingStrategy,
        user: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Score and rank models based on strategy.
        
        Args:
            models: List of available models
            analysis: Prompt analysis
            strategy: Routing strategy
            user: User information
        
        Returns:
            Ranked list of models with scores
        """
        scored_models = []
        
        for model in models:
            # Calculate base score from strategy
            score = await strategy.calculate_score(
                model=model,
                analysis=analysis,
                user=user
            )
            
            # Apply penalties/boosts
            score = self._apply_capability_boost(score, model, analysis)
            score = self._apply_latency_penalty(score, model)
            score = self._apply_cost_penalty(score, model, user)
            score = self._apply_reliability_boost(score, model)
            
            # Skip models below confidence threshold
            if score >= self.config.min_confidence:
                scored_models.append({
                    "model_id": model["id"],
                    "model_name": model.get("name", model["id"]),
                    "provider": model.get("provider", "unknown"),
                    "score": score,
                    "capabilities": model.get("capabilities", []),
                    "latency_ms": model.get("latency_p95_ms", 500),
                    "cost_per_token": model.get("cost_per_token", 0),
                    "status": model.get("status", "unknown")
                })
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_models[:self.config.max_candidates]
    
    def _apply_capability_boost(
        self,
        score: float,
        model: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> float:
        """Boost score if model has required capabilities."""
        prompt_type = analysis.get("prompt_type", "general")
        capabilities = model.get("capabilities", [])
        
        # Check if model has capability for this prompt type
        if prompt_type in capabilities:
            score += 0.2
        elif "general" in capabilities:
            score += 0.1
        
        # Boost for specialized models
        if prompt_type == "code" and "code" in capabilities:
            score += 0.3
        elif prompt_type == "creative" and "creative" in capabilities:
            score += 0.3
        elif prompt_type == "reasoning" and "reasoning" in capabilities:
            score += 0.3
        
        return min(score, 1.0)
    
    def _apply_latency_penalty(
        self,
        score: float,
        model: Dict[str, Any]
    ) -> float:
        """Apply penalty for high-latency models."""
        latency = model.get("latency_p95_ms", 500)
        
        if latency > self.config.max_latency_ms:
            score *= 0.5
        elif latency > self.config.max_latency_ms * 0.7:
            score *= 0.8
        
        return score
    
    def _apply_cost_penalty(
        self,
        score: float,
        model: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None
    ) -> float:
        """Apply penalty for expensive models."""
        cost = model.get("cost_per_token", 0)
        
        # Free tier users get higher cost penalty
        if user and user.get("tier") == "free":
            if cost > 0.00001:  # $0.00001 per token
                score *= 0.6
            elif cost > 0.000001:
                score *= 0.8
        else:
            if cost > self.config.max_cost_usd:
                score *= 0.7
        
        return score
    
    def _apply_reliability_boost(
        self,
        score: float,
        model: Dict[str, Any]
    ) -> float:
        """Boost score for reliable models."""
        error_rate = model.get("error_rate", 0)
        
        if error_rate < 0.01:  # <1% error rate
            score += 0.1
        elif error_rate > 0.05:  # >5% error rate
            score *= 0.8
        
        return min(score, 1.0)
    
    # ========================================================================
    # MODEL FILTERING
    # ========================================================================
    
    async def _get_available_models(
        self,
        user: Optional[Dict[str, Any]] = None,
        analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get list of available models for routing."""
        all_models = await self.model_registry.get_all_models()
        available_models = []
        
        for model in all_models:
            # Check if model is available
            if not self._is_model_available(model):
                continue
            
            # Check user tier access
            if not self._check_user_access(model, user):
                continue
            
            # Check if model has required capabilities
            if analysis and not self._check_capabilities(model, analysis):
                continue
            
            # Get real-time status and metrics
            model_id = model["id"]
            
            # Check if model is loaded (for local models)
            if model.get("type") != "external":
                is_loaded = await self.model_manager.is_loaded(model_id)
                if not is_loaded:
                    # Can still route, but add loading penalty
                    model["load_penalty"] = 0.3
                else:
                    model["load_penalty"] = 0
            
            # Add to available models
            available_models.append(model)
        
        return available_models
    
    def _is_model_available(self, model: Dict[str, Any]) -> bool:
        """Check if model is available for inference."""
        status = model.get("status", "unknown")
        
        if status == "available":
            return True
        
        if status == "ready":
            return True
        
        if model.get("type") == "external" and status == "available":
            return True
        
        return False
    
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
        access_tier = model.get("access_tier", "free")
        
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
    
    # ========================================================================
    # CUSTOM RULES
    # ========================================================================
    
    def _load_custom_rules(self):
        """Load custom routing rules from configuration."""
        try:
            rules_config = settings.routing_config_dict
            self.custom_rules = rules_config.get("rules", [])
            
            logger.info(
                "custom_rules_loaded",
                rule_count=len(self.custom_rules)
            )
        except Exception as e:
            logger.error(
                "failed_to_load_custom_rules",
                error=str(e)
            )
            self.custom_rules = []
    
    async def _apply_custom_rules(
        self,
        prompt: str,
        analysis: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Apply custom routing rules."""
        for rule in self.custom_rules:
            if not rule.get("enabled", True):
                continue
            
            if await self._evaluate_rule(rule, prompt, analysis, user):
                logger.debug(
                    "custom_rule_matched",
                    rule=rule.get("name"),
                    priority=rule.get("priority")
                )
                
                return {
                    "selected_model": rule["action"]["model"],
                    "strategy": "custom_rule",
                    "confidence": 1.0,
                    "reasoning": f"Custom rule matched: {rule.get('name', 'Unnamed')}",
                    "rule": rule.get("name"),
                    "alternatives": []
                }
        
        return None
    
    async def _evaluate_rule(
        self,
        rule: Dict[str, Any],
        prompt: str,
        analysis: Dict[str, Any],
        user: Optional[Dict[str, Any]]
    ) -> bool:
        """Evaluate if a custom rule matches."""
        conditions = rule.get("conditions", [])
        
        for condition in conditions:
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            
            # Get field value
            if field == "prompt_length":
                field_value = len(prompt)
            elif field == "prompt_type":
                field_value = analysis.get("prompt_type")
            elif field == "complexity":
                field_value = analysis.get("complexity_score")
            elif field == "user_tier":
                field_value = user.get("tier") if user else "anonymous"
            elif field == "user_id":
                field_value = user.get("id") if user else None
            else:
                field_value = analysis.get(field)
            
            # Evaluate operator
            if operator == "eq":
                if field_value != value:
                    return False
            elif operator == "neq":
                if field_value == value:
                    return False
            elif operator == "gt":
                if not (field_value and field_value > value):
                    return False
            elif operator == "gte":
                if not (field_value and field_value >= value):
                    return False
            elif operator == "lt":
                if not (field_value and field_value < value):
                    return False
            elif operator == "lte":
                if not (field_value and field_value <= value):
                    return False
            elif operator == "contains":
                if value not in str(field_value):
                    return False
            elif operator == "in":
                if field_value not in value:
                    return False
        
        return True
    
    # ========================================================================
    # CACHING
    # ========================================================================
    
    def _generate_cache_key(
        self,
        prompt_hash: str,
        strategy: Optional[str],
        user: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for routing decision."""
        key_parts = [
            "routing",
            prompt_hash,
            strategy or self.config.default_strategy
        ]
        
        if user:
            key_parts.append(user.get("tier", "free"))
            if user.get("is_admin"):
                key_parts.append("admin")
        
        return ":".join(key_parts)
    
    async def _get_cached_decision(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached routing decision."""
        try:
            cached = await self.cache_manager.get(cache_key)
            if cached:
                cached["cache_hit"] = True
                return cached
        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
        
        return None
    
    async def _cache_decision(self, cache_key: str, decision: Dict[str, Any]):
        """Cache routing decision."""
        try:
            await self.cache_manager.set(
                key=cache_key,
                value=decision,
                ttl=self.config.cache_ttl
            )
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")
    
    async def clear_cache(self) -> int:
        """Clear routing decision cache."""
        try:
            pattern = "routing:*"
            keys = await self.cache_manager.keys(pattern)
            if keys:
                await self.cache_manager.delete(*keys)
                return len(keys)
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
        
        return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            pattern = "routing:*"
            keys = await self.cache_manager.keys(pattern)
            
            return {
                "entries": len(keys),
                "size_bytes": sum(len(k) for k in keys) if keys else 0,
                "hits": 0,  # TODO: Track hits/misses
                "misses": 0
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                "entries": 0,
                "size_bytes": 0,
                "hits": 0,
                "misses": 0
            }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
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
        
        reasoning_parts = [
            f"Selected model: {model_name}",
            f"Strategy: {strategy}",
            f"Confidence: {score:.2f}",
            f"Prompt type: {prompt_type}"
        ]
        
        # Add specific reasoning based on strategy
        if strategy == "latency":
            latency = selected_model.get("latency_ms", "unknown")
            reasoning_parts.append(f"Expected latency: {latency}ms")
        elif strategy == "cost":
            cost = selected_model.get("cost_per_token", 0)
            reasoning_parts.append(f"Cost per token: ${cost:.6f}")
        elif strategy == "quality":
            reasoning_parts.append("Optimized for response quality")
        elif strategy == "hybrid":
            reasoning_parts.append("Balanced optimization across latency, cost, and quality")
        
        # Add capability match if applicable
        if prompt_type in selected_model.get("capabilities", []):
            reasoning_parts.append(f"Specialized in {prompt_type} tasks")
        
        return " | ".join(reasoning_parts)
    
    async def _record_metrics(self, decision: RoutingDecision):
        """Record routing decision metrics."""
        await self.metrics_collector.record_routing_decision(
            decision.to_dict()
        )
    
    async def reload_rules(self):
        """Reload custom routing rules from configuration."""
        self._load_custom_rules()
        await self.clear_cache()
        
        logger.info(
            "routing_rules_reloaded",
            rule_count=len(self.custom_rules)
        )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_prompt_router = None


def get_prompt_router() -> PromptRouter:
    """Get singleton prompt router instance."""
    global _prompt_router
    if not _prompt_router:
        _prompt_router = PromptRouter()
    return _prompt_router


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "PromptRouter",
    "RoutingDecision",
    "RoutingConfig",
    "get_prompt_router"
]