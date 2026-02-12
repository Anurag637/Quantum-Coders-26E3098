"""
Routing Strategies - Production Ready
Implementation of various routing algorithms for model selection based on
latency, cost, quality, load balancing, and hybrid approaches.
"""

import random
import math
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod

from core.logging import get_logger
from config import settings

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# BASE ROUTING STRATEGY
# ============================================================================

class RoutingStrategy(ABC):
    """Base class for all routing strategies."""
    
    @abstractmethod
    async def calculate_score(
        self,
        model: Dict[str, Any],
        analysis: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate suitability score for a model (0-1).
        
        Args:
            model: Model metadata
            analysis: Prompt analysis results
            user: User information
        
        Returns:
            Score between 0 and 1, higher is better
        """
        pass
    
    @abstractmethod
    async def make_decision(
        self,
        prompt: str,
        analysis: Dict[str, Any],
        available_models: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make routing decision based on strategy.
        
        Args:
            prompt: Input prompt
            analysis: Prompt analysis results
            available_models: List of available models
            context: Additional context
        
        Returns:
            Routing decision with selected model and metadata
        """
        pass
    
    def normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to 0-1 range."""
        if max_val == min_val:
            return 1.0
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
    
    def get_model_metadata(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standardized metadata from model."""
        return {
            "model_id": model["id"],
            "model_name": model.get("name", model["id"]),
            "provider": model.get("provider", "unknown"),
            "type": model.get("type", "unknown"),
            "latency_ms": model.get("latency_p95_ms", 500),
            "cost_per_token": model.get("cost_per_token", 0),
            "quality_score": model.get("quality_score", 0.7),
            "reliability": 1 - model.get("error_rate", 0),
            "capabilities": model.get("capabilities", []),
            "load_penalty": model.get("load_penalty", 0)
        }


# ============================================================================
# LATENCY OPTIMIZED STRATEGY
# ============================================================================

class LatencyOptimizedStrategy(RoutingStrategy):
    """
    Optimize for lowest inference latency.
    
    Use Case: Real-time applications, interactive chat, user-facing features
    Priority: Speed > Cost > Quality
    """
    
    def __init__(self):
        self.max_latency_weight = 0.7
        self.load_penalty_weight = 0.2
        self.capability_boost = 0.1
    
    async def calculate_score(
        self,
        model: Dict[str, Any],
        analysis: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None
    ) -> float:
        metadata = self.get_model_metadata(model)
        
        # Base score from latency (lower is better)
        latency_ms = metadata["latency_ms"]
        
        # Normalize latency (assume range 100-2000ms)
        normalized_latency = self.normalize_score(
            latency_ms,
            min_val=100,
            max_val=2000
        )
        latency_score = 1 - normalized_latency  # Invert: lower latency = higher score
        
        # Apply load penalty
        load_penalty = metadata.get("load_penalty", 0)
        load_score = 1 - load_penalty
        
        # Check capability match
        capability_boost = 0
        prompt_type = analysis.get("prompt_type", "general")
        if prompt_type in metadata["capabilities"]:
            capability_boost = self.capability_boost
        
        # Calculate weighted score
        score = (
            latency_score * self.max_latency_weight +
            load_score * self.load_penalty_weight +
            capability_boost
        )
        
        # External APIs tend to have higher latency, apply penalty
        if metadata["type"] == "external":
            score *= 0.9
        
        return min(score, 1.0)
    
    async def make_decision(
        self,
        prompt: str,
        analysis: Dict[str, Any],
        available_models: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # Score all models
        scored_models = []
        for model in available_models:
            score = await self.calculate_score(model, analysis)
            scored_models.append((score, model))
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_models:
            return {
                "selected_model": None,
                "strategy": "latency",
                "confidence": 0,
                "reasoning": "No available models",
                "alternatives": []
            }
        
        best_score, best_model = scored_models[0]
        
        # Get alternatives
        alternatives = []
        for score, model in scored_models[1:4]:
            metadata = self.get_model_metadata(model)
            alternatives.append({
                "model_id": metadata["model_id"],
                "model_name": metadata["model_name"],
                "score": score,
                "latency_ms": metadata["latency_ms"]
            })
        
        # Generate reasoning
        metadata = self.get_model_metadata(best_model)
        reasoning = (
            f"Selected {metadata['model_name']} for lowest latency "
            f"({metadata['latency_ms']}ms). "
            f"Confidence: {best_score:.2f}."
        )
        
        return {
            "selected_model": metadata["model_id"],
            "strategy": "latency",
            "confidence": best_score,
            "reasoning": reasoning,
            "alternatives": alternatives
        }


# ============================================================================
# COST OPTIMIZED STRATEGY
# ============================================================================

class CostOptimizedStrategy(RoutingStrategy):
    """
    Optimize for lowest cost per request.
    
    Use Case: Batch processing, background jobs, high-volume applications
    Priority: Cost > Speed > Quality
    """
    
    def __init__(self):
        self.cost_weight = 0.6
        self.quality_penalty_threshold = 0.5
        self.free_tier_bonus = 0.2
    
    async def calculate_score(
        self,
        model: Dict[str, Any],
        analysis: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None
    ) -> float:
        metadata = self.get_model_metadata(model)
        
        # Base score from cost (lower is better)
        cost_per_token = metadata["cost_per_token"]
        
        # Normalize cost (assume range $0 - $0.0001 per token)
        normalized_cost = self.normalize_score(
            cost_per_token,
            min_val=0,
            max_val=0.0001
        )
        cost_score = 1 - normalized_cost  # Invert: lower cost = higher score
        
        # Free tier bonus
        if user and user.get("tier") == "free":
            if cost_per_token == 0:
                cost_score += self.free_tier_bonus
        
        # Quality penalty for very cheap models
        quality_score = metadata["quality_score"]
        if quality_score < self.quality_penalty_threshold:
            cost_score *= 0.8
        
        # Local models are cheaper than external APIs
        if metadata["type"] != "external":
            cost_score += 0.1
        
        return min(cost_score, 1.0)
    
    async def make_decision(
        self,
        prompt: str,
        analysis: Dict[str, Any],
        available_models: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # Score all models
        scored_models = []
        for model in available_models:
            score = await self.calculate_score(model, analysis, context.get("user") if context else None)
            scored_models.append((score, model))
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_models:
            return {
                "selected_model": None,
                "strategy": "cost",
                "confidence": 0,
                "reasoning": "No available models",
                "alternatives": []
            }
        
        best_score, best_model = scored_models[0]
        
        # Get alternatives
        alternatives = []
        for score, model in scored_models[1:4]:
            metadata = self.get_model_metadata(model)
            alternatives.append({
                "model_id": metadata["model_id"],
                "model_name": metadata["model_name"],
                "score": score,
                "cost_per_token": metadata["cost_per_token"]
            })
        
        # Generate reasoning
        metadata = self.get_model_metadata(best_model)
        cost_per_1k = metadata["cost_per_token"] * 1000
        
        reasoning = (
            f"Selected {metadata['model_name']} for lowest cost "
            f"(${cost_per_1k:.4f}/1K tokens). "
            f"Confidence: {best_score:.2f}."
        )
        
        return {
            "selected_model": metadata["model_id"],
            "strategy": "cost",
            "confidence": best_score,
            "reasoning": reasoning,
            "alternatives": alternatives
        }


# ============================================================================
# QUALITY OPTIMIZED STRATEGY
# ============================================================================

class QualityOptimizedStrategy(RoutingStrategy):
    """
    Optimize for highest response quality.
    
    Use Case: Creative writing, complex reasoning, critical applications
    Priority: Quality > Speed > Cost
    """
    
    def __init__(self):
        self.quality_weight = 0.7
        self.capability_weight = 0.2
        self.reliability_weight = 0.1
    
    async def calculate_score(
        self,
        model: Dict[str, Any],
        analysis: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None
    ) -> float:
        metadata = self.get_model_metadata(model)
        
        # Base quality score
        quality_score = metadata["quality_score"]
        
        # Capability match boost
        capability_boost = 0
        prompt_type = analysis.get("prompt_type", "general")
        if prompt_type in metadata["capabilities"]:
            capability_boost = 0.2
        elif "general" in metadata["capabilities"]:
            capability_boost = 0.1
        
        # Reliability score
        reliability_score = metadata["reliability"]
        
        # Calculate weighted score
        score = (
            quality_score * self.quality_weight +
            capability_boost * self.capability_weight +
            reliability_score * self.reliability_weight
        )
        
        # Larger models tend to have higher quality
        if "70b" in metadata["model_id"].lower() or "gpt-4" in metadata["model_id"].lower():
            score += 0.1
        
        return min(score, 1.0)
    
    async def make_decision(
        self,
        prompt: str,
        analysis: Dict[str, Any],
        available_models: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # Score all models
        scored_models = []
        for model in available_models:
            score = await self.calculate_score(model, analysis)
            scored_models.append((score, model))
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_models:
            return {
                "selected_model": None,
                "strategy": "quality",
                "confidence": 0,
                "reasoning": "No available models",
                "alternatives": []
            }
        
        best_score, best_model = scored_models[0]
        
        # Get alternatives
        alternatives = []
        for score, model in scored_models[1:4]:
            metadata = self.get_model_metadata(model)
            alternatives.append({
                "model_id": metadata["model_id"],
                "model_name": metadata["model_name"],
                "score": score,
                "quality_score": metadata["quality_score"]
            })
        
        # Generate reasoning
        metadata = self.get_model_metadata(best_model)
        
        reasoning = (
            f"Selected {metadata['model_name']} for highest quality "
            f"(quality score: {metadata['quality_score']:.2f}). "
            f"Confidence: {best_score:.2f}."
        )
        
        return {
            "selected_model": metadata["model_id"],
            "strategy": "quality",
            "confidence": best_score,
            "reasoning": reasoning,
            "alternatives": alternatives
        }


# ============================================================================
# HYBRID STRATEGY
# ============================================================================

class HybridStrategy(RoutingStrategy):
    """
    Balanced optimization across latency, cost, and quality.
    
    Use Case: General purpose, mixed workloads, default strategy
    Priority: Balanced based on configurable weights
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "latency": 0.4,
            "cost": 0.3,
            "quality": 0.3
        }
    
    async def calculate_score(
        self,
        model: Dict[str, Any],
        analysis: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None
    ) -> float:
        metadata = self.get_model_metadata(model)
        
        # Latency score (lower is better)
        normalized_latency = self.normalize_score(
            metadata["latency_ms"],
            min_val=100,
            max_val=2000
        )
        latency_score = 1 - normalized_latency
        
        # Cost score (lower is better)
        normalized_cost = self.normalize_score(
            metadata["cost_per_token"],
            min_val=0,
            max_val=0.0001
        )
        cost_score = 1 - normalized_cost
        
        # Quality score (higher is better)
        quality_score = metadata["quality_score"]
        
        # Capability match boost
        prompt_type = analysis.get("prompt_type", "general")
        if prompt_type in metadata["capabilities"]:
            quality_score += 0.1
        
        # Calculate weighted score
        score = (
            latency_score * self.weights["latency"] +
            cost_score * self.weights["cost"] +
            quality_score * self.weights["quality"]
        )
        
        return min(score, 1.0)
    
    async def make_decision(
        self,
        prompt: str,
        analysis: Dict[str, Any],
        available_models: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # Score all models
        scored_models = []
        for model in available_models:
            score = await self.calculate_score(model, analysis, context.get("user") if context else None)
            scored_models.append((score, model))
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_models:
            return {
                "selected_model": None,
                "strategy": "hybrid",
                "confidence": 0,
                "reasoning": "No available models",
                "alternatives": []
            }
        
        best_score, best_model = scored_models[0]
        
        # Get alternatives
        alternatives = []
        for score, model in scored_models[1:4]:
            metadata = self.get_model_metadata(model)
            alternatives.append({
                "model_id": metadata["model_id"],
                "model_name": metadata["model_name"],
                "score": score,
                "latency_ms": metadata["latency_ms"],
                "cost_per_token": metadata["cost_per_token"],
                "quality_score": metadata["quality_score"]
            })
        
        # Generate reasoning
        metadata = self.get_model_metadata(best_model)
        
        reasoning = (
            f"Selected {metadata['model_name']} for balanced performance. "
            f"Weights: latency={self.weights['latency']}, "
            f"cost={self.weights['cost']}, "
            f"quality={self.weights['quality']}. "
            f"Confidence: {best_score:.2f}."
        )
        
        return {
            "selected_model": metadata["model_id"],
            "strategy": "hybrid",
            "confidence": best_score,
            "reasoning": reasoning,
            "alternatives": alternatives
        }


# ============================================================================
# ROUND ROBIN STRATEGY
# ============================================================================

class RoundRobinStrategy(RoutingStrategy):
    """
    Distribute requests evenly across available models.
    
    Use Case: Load testing, A/B testing, fair resource allocation
    Priority: Distribution > Performance
    """
    
    def __init__(self):
        self.counter = 0
    
    async def calculate_score(
        self,
        model: Dict[str, Any],
        analysis: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None
    ) -> float:
        # Not used for round robin
        return 0.5
    
    async def make_decision(
        self,
        prompt: str,
        analysis: Dict[str, Any],
        available_models: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not available_models:
            return {
                "selected_model": None,
                "strategy": "round_robin",
                "confidence": 0,
                "reasoning": "No available models",
                "alternatives": []
            }
        
        # Simple round robin selection
        index = self.counter % len(available_models)
        selected_model = available_models[index]
        self.counter += 1
        
        metadata = self.get_model_metadata(selected_model)
        
        # Get alternatives (all other models)
        alternatives = []
        for i, model in enumerate(available_models):
            if i != index:
                alt_metadata = self.get_model_metadata(model)
                alternatives.append({
                    "model_id": alt_metadata["model_id"],
                    "model_name": alt_metadata["model_name"]
                })
        
        reasoning = (
            f"Selected {metadata['model_name']} via round-robin distribution "
            f"(request #{self.counter})."
        )
        
        return {
            "selected_model": metadata["model_id"],
            "strategy": "round_robin",
            "confidence": 0.5,
            "reasoning": reasoning,
            "alternatives": alternatives[:3]
        }


# ============================================================================
# LEAST CONNECTIONS STRATEGY
# ============================================================================

class LeastConnectionsStrategy(RoutingStrategy):
    """
    Route to model with fewest active connections.
    
    Use Case: High concurrency, variable load patterns
    Priority: Load distribution > Individual performance
    """
    
    def __init__(self):
        self.connection_counts = {}
    
    async def calculate_score(
        self,
        model: Dict[str, Any],
        analysis: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None
    ) -> float:
        # Score based on active connections (lower is better)
        model_id = model["id"]
        active_connections = self.connection_counts.get(model_id, 0)
        
        # Normalize (assume 0-50 connections)
        normalized = self.normalize_score(
            active_connections,
            min_val=0,
            max_val=50
        )
        
        # Invert: fewer connections = higher score
        return 1 - normalized
    
    async def make_decision(
        self,
        prompt: str,
        analysis: Dict[str, Any],
        available_models: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not available_models:
            return {
                "selected_model": None,
                "strategy": "least_connections",
                "confidence": 0,
                "reasoning": "No available models",
                "alternatives": []
            }
        
        # Score all models
        scored_models = []
        for model in available_models:
            score = await self.calculate_score(model, analysis)
            scored_models.append((score, model))
        
        # Sort by score descending (fewest connections first)
        scored_models.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_model = scored_models[0]
        metadata = self.get_model_metadata(best_model)
        
        # Increment connection count for selected model
        model_id = metadata["model_id"]
        self.connection_counts[model_id] = self.connection_counts.get(model_id, 0) + 1
        
        # Get alternatives
        alternatives = []
        for score, model in scored_models[1:4]:
            alt_metadata = self.get_model_metadata(model)
            alt_id = alt_metadata["model_id"]
            alternatives.append({
                "model_id": alt_id,
                "model_name": alt_metadata["model_name"],
                "active_connections": self.connection_counts.get(alt_id, 0)
            })
        
        reasoning = (
            f"Selected {metadata['model_name']} with fewest active connections "
            f"({self.connection_counts.get(model_id, 0)}). "
            f"Confidence: {best_score:.2f}."
        )
        
        return {
            "selected_model": metadata["model_id"],
            "strategy": "least_connections",
            "confidence": best_score,
            "reasoning": reasoning,
            "alternatives": alternatives
        }
    
    def release_connection(self, model_id: str):
        """Release a connection when request completes."""
        if model_id in self.connection_counts:
            self.connection_counts[model_id] = max(0, self.connection_counts[model_id] - 1)


# ============================================================================
# ADAPTIVE STRATEGY
# ============================================================================

class AdaptiveStrategy(RoutingStrategy):
    """
    Dynamically adjust weights based on performance history.
    
    Use Case: Self-optimizing systems, learning from feedback
    Priority: Adapt to changing conditions
    """
    
    def __init__(self, adaptation_rate: float = 0.1):
        self.adaptation_rate = adaptation_rate
        self.performance_history = {}
        self.base_weights = {
            "latency": 0.4,
            "cost": 0.3,
            "quality": 0.3
        }
    
    async def calculate_score(
        self,
        model: Dict[str, Any],
        analysis: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None
    ) -> float:
        metadata = self.get_model_metadata(model)
        model_id = metadata["model_id"]
        
        # Get adaptive weights for this model
        weights = self._get_adaptive_weights(model_id)
        
        # Calculate component scores
        normalized_latency = self.normalize_score(
            metadata["latency_ms"],
            min_val=100,
            max_val=2000
        )
        latency_score = 1 - normalized_latency
        
        normalized_cost = self.normalize_score(
            metadata["cost_per_token"],
            min_val=0,
            max_val=0.0001
        )
        cost_score = 1 - normalized_cost
        
        quality_score = metadata["quality_score"]
        
        # Calculate weighted score
        score = (
            latency_score * weights.get("latency", 0.33) +
            cost_score * weights.get("cost", 0.33) +
            quality_score * weights.get("quality", 0.33)
        )
        
        return min(score, 1.0)
    
    async def make_decision(
        self,
        prompt: str,
        analysis: Dict[str, Any],
        available_models: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # Score all models
        scored_models = []
        for model in available_models:
            score = await self.calculate_score(model, analysis, context.get("user") if context else None)
            scored_models.append((score, model))
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_models:
            return {
                "selected_model": None,
                "strategy": "adaptive",
                "confidence": 0,
                "reasoning": "No available models",
                "alternatives": []
            }
        
        best_score, best_model = scored_models[0]
        metadata = self.get_model_metadata(best_model)
        
        # Get alternatives
        alternatives = []
        for score, model in scored_models[1:4]:
            alt_metadata = self.get_model_metadata(model)
            alternatives.append({
                "model_id": alt_metadata["model_id"],
                "model_name": alt_metadata["model_name"],
                "score": score
            })
        
        # Generate reasoning
        model_id = metadata["model_id"]
        weights = self._get_adaptive_weights(model_id)
        
        reasoning = (
            f"Selected {metadata['model_name']} with adaptive strategy. "
            f"Weights: latency={weights['latency']:.2f}, "
            f"cost={weights['cost']:.2f}, "
            f"quality={weights['quality']:.2f}. "
            f"Confidence: {best_score:.2f}."
        )
        
        return {
            "selected_model": metadata["model_id"],
            "strategy": "adaptive",
            "confidence": best_score,
            "reasoning": reasoning,
            "alternatives": alternatives
        }
    
    def _get_adaptive_weights(self, model_id: str) -> Dict[str, float]:
        """Get adaptive weights for a model based on performance history."""
        if model_id not in self.performance_history:
            return self.base_weights.copy()
        
        history = self.performance_history[model_id]
        
        # Calculate performance scores
        avg_latency = history.get("avg_latency", 500)
        avg_cost = history.get("avg_cost", 0.00001)
        avg_quality = history.get("avg_quality", 0.7)
        
        # Normalize performance metrics
        latency_perf = 1 - self.normalize_score(avg_latency, 100, 2000)
        cost_perf = 1 - self.normalize_score(avg_cost, 0, 0.0001)
        quality_perf = avg_quality
        
        # Adjust weights based on performance
        weights = self.base_weights.copy()
        
        weights["latency"] = self.base_weights["latency"] * (1 + (latency_perf - 0.5) * self.adaptation_rate)
        weights["cost"] = self.base_weights["cost"] * (1 + (cost_perf - 0.5) * self.adaptation_rate)
        weights["quality"] = self.base_weights["quality"] * (1 + (quality_perf - 0.5) * self.adaptation_rate)
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total
        
        return weights
    
    def update_performance(
        self,
        model_id: str,
        latency_ms: float,
        cost: float,
        quality_score: float
    ):
        """Update performance history for a model."""
        if model_id not in self.performance_history:
            self.performance_history[model_id] = {
                "latency_samples": [],
                "cost_samples": [],
                "quality_samples": [],
                "avg_latency": latency_ms,
                "avg_cost": cost,
                "avg_quality": quality_score
            }
        
        history = self.performance_history[model_id]
        
        # Add samples (keep last 100)
        history["latency_samples"].append(latency_ms)
        history["cost_samples"].append(cost)
        history["quality_samples"].append(quality_score)
        
        if len(history["latency_samples"]) > 100:
            history["latency_samples"].pop(0)
            history["cost_samples"].pop(0)
            history["quality_samples"].pop(0)
        
        # Update averages
        history["avg_latency"] = sum(history["latency_samples"]) / len(history["latency_samples"])
        history["avg_cost"] = sum(history["cost_samples"]) / len(history["cost_samples"])
        history["avg_quality"] = sum(history["quality_samples"]) / len(history["quality_samples"])


# ============================================================================
# STRATEGY FACTORY
# ============================================================================

class RoutingStrategyFactory:
    """Factory for creating routing strategy instances."""
    
    def __init__(self):
        self.strategies = {
            "latency": LatencyOptimizedStrategy,
            "cost": CostOptimizedStrategy,
            "quality": QualityOptimizedStrategy,
            "hybrid": HybridStrategy,
            "round_robin": RoundRobinStrategy,
            "least_connections": LeastConnectionsStrategy,
            "adaptive": AdaptiveStrategy
        }
    
    def get_strategy(self, name: str, **kwargs) -> RoutingStrategy:
        """
        Get a routing strategy instance by name.
        
        Args:
            name: Strategy name
            **kwargs: Strategy-specific parameters
        
        Returns:
            RoutingStrategy instance
        
        Raises:
            ValueError: If strategy name is unknown
        """
        if name not in self.strategies:
            raise ValueError(f"Unknown routing strategy: {name}")
        
        strategy_class = self.strategies[name]
        
        if name == "hybrid":
            weights = kwargs.get("weights", settings.routing_weights)
            return strategy_class(weights=weights)
        elif name == "adaptive":
            rate = kwargs.get("adaptation_rate", 0.1)
            return strategy_class(adaptation_rate=rate)
        else:
            return strategy_class()
    
    def list_strategies(self) -> List[str]:
        """List all available strategy names."""
        return list(self.strategies.keys())


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "RoutingStrategy",
    "LatencyOptimizedStrategy",
    "CostOptimizedStrategy",
    "QualityOptimizedStrategy",
    "HybridStrategy",
    "RoundRobinStrategy",
    "LeastConnectionsStrategy",
    "AdaptiveStrategy",
    "RoutingStrategyFactory"
]