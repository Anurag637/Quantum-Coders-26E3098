from dataclasses import dataclass
from typing import Optional


# =========================================================
# Decision Input
# =========================================================

@dataclass
class DecisionContext:
    intent: str
    complexity: str
    estimated_tokens: int
    user_tier: str
    health_status: float


# =========================================================
# Decision Result
# =========================================================

@dataclass
class DecisionResult:
    selected_tier: str
    fallback_tier: Optional[str]
    reason: str


# =========================================================
# Decision Tree Engine
# =========================================================

class DecisionTreeRouter:
    """
    Deterministic rule-based routing.
    Fast and explainable.
    """

    HIGH_TOKEN_THRESHOLD = 6000
    MEDIUM_TOKEN_THRESHOLD = 2000

    # =====================================================
    # Main Decision
    # =====================================================

    def decide(self, context: DecisionContext) -> DecisionResult:

        # ---------------------------------
        # 1. Health Check First
        # ---------------------------------
        if context.health_status < 0.5:
            return DecisionResult(
                selected_tier="medium",
                fallback_tier="small",
                reason="Primary tier unhealthy"
            )

        # ---------------------------------
        # 2. Enterprise Override
        # ---------------------------------
        if context.user_tier == "enterprise":
            if context.complexity == "high":
                return DecisionResult(
                    selected_tier="large",
                    fallback_tier="medium",
                    reason="Enterprise high complexity"
                )
            return DecisionResult(
                selected_tier="medium",
                fallback_tier="small",
                reason="Enterprise default routing"
            )

        # ---------------------------------
        # 3. Code Intent
        # ---------------------------------
        if context.intent == "code_generation":
            return DecisionResult(
                selected_tier="coder",
                fallback_tier="medium",
                reason="Code generation task"
            )

        # ---------------------------------
        # 4. High Complexity
        # ---------------------------------
        if context.complexity == "high":
            return DecisionResult(
                selected_tier="large",
                fallback_tier="medium",
                reason="High complexity detected"
            )

        # ---------------------------------
        # 5. Token-Based Decision
        # ---------------------------------
        if context.estimated_tokens > self.HIGH_TOKEN_THRESHOLD:
            return DecisionResult(
                selected_tier="large",
                fallback_tier="medium",
                reason="High token requirement"
            )

        if context.estimated_tokens > self.MEDIUM_TOKEN_THRESHOLD:
            return DecisionResult(
                selected_tier="medium",
                fallback_tier="small",
                reason="Medium token requirement"
            )

        # ---------------------------------
        # 6. Default
        # ---------------------------------
        return DecisionResult(
            selected_tier="small",
            fallback_tier=None,
            reason="Default small tier routing"
        )