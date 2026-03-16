from dataclasses import dataclass
from typing import List, Dict, Optional


# =========================================================
# Model Profile (Input to Scoring)
# =========================================================

@dataclass
class ModelProfile:
    name: str
    tier: str
    supports_code: bool
    supports_creative: bool
    avg_latency_ms: float
    cost_per_1k_tokens: float
    health_score: float        # 0.0 - 1.0
    current_load: float        # 0.0 - 1.0


# =========================================================
# Routing Context
# =========================================================

@dataclass
class RoutingContext:
    intent: str
    complexity: str
    required_tier: str
    estimated_tokens: int


# =========================================================
# Scoring Engine
# =========================================================

class ScoringEngine:
    """
    Computes a weighted score for each model.
    Higher score = better candidate.
    """

    # Weight configuration
    WEIGHTS = {
        "capability": 0.30,
        "latency": 0.20,
        "cost": 0.20,
        "health": 0.20,
        "load": 0.10,
    }

    # =====================================================
    # Public API
    # =====================================================

    def score_models(
        self,
        models: List[ModelProfile],
        context: RoutingContext,
    ) -> Optional[ModelProfile]:

        best_score = -1
        best_model = None

        for model in models:
            score = self._score_model(model, context)

            if score > best_score:
                best_score = score
                best_model = model

        return best_model

    # =====================================================
    # Core Scoring Logic
    # =====================================================

    def _score_model(
        self,
        model: ModelProfile,
        context: RoutingContext,
    ) -> float:

        capability_score = self._capability_score(model, context)
        latency_score = self._latency_score(model)
        cost_score = self._cost_score(model, context.estimated_tokens)
        health_score = model.health_score
        load_score = 1.0 - model.current_load

        total_score = (
            capability_score * self.WEIGHTS["capability"] +
            latency_score * self.WEIGHTS["latency"] +
            cost_score * self.WEIGHTS["cost"] +
            health_score * self.WEIGHTS["health"] +
            load_score * self.WEIGHTS["load"]
        )

        return round(total_score, 4)

    # =====================================================
    # Individual Scoring Components
    # =====================================================

    def _capability_score(
        self,
        model: ModelProfile,
        context: RoutingContext,
    ) -> float:

        score = 0.5

        # Tier match
        if model.tier == context.required_tier:
            score += 0.3

        # Intent match
        if context.intent == "code_generation" and model.supports_code:
            score += 0.2

        if context.intent == "creative_writing" and model.supports_creative:
            score += 0.2

        return min(score, 1.0)

    def _latency_score(self, model: ModelProfile) -> float:
        # Normalize: lower latency = higher score
        return 1 / (1 + model.avg_latency_ms / 1000)

    def _cost_score(
        self,
        model: ModelProfile,
        estimated_tokens: int,
    ) -> float:

        estimated_cost = (estimated_tokens / 1000) * model.cost_per_1k_tokens

        # Lower cost = higher score
        return 1 / (1 + estimated_cost)