from dataclasses import dataclass
from typing import Optional


# =========================================================
# Usage Input
# =========================================================

@dataclass
class UsageInput:
    model_name: str
    input_tokens: int
    output_tokens: int
    user_tier: str
    input_cost_per_1k: float
    output_cost_per_1k: float


# =========================================================
# Usage Result
# =========================================================

@dataclass
class UsageResult:
    total_tokens: int
    base_cost: float
    tier_multiplier: float
    final_cost: float


# =========================================================
# Usage Calculator
# =========================================================

class UsageCalculator:
    """
    Computes token usage and billing cost.
    """

    TIER_MULTIPLIERS = {
        "free": 1.0,
        "pro": 0.95,         # small discount
        "enterprise": 0.85   # volume discount
    }

    # =====================================================
    # Main Calculation
    # =====================================================

    def calculate(self, usage: UsageInput) -> UsageResult:

        total_tokens = usage.input_tokens + usage.output_tokens

        input_cost = (usage.input_tokens / 1000) * usage.input_cost_per_1k
        output_cost = (usage.output_tokens / 1000) * usage.output_cost_per_1k

        base_cost = input_cost + output_cost

        multiplier = self.TIER_MULTIPLIERS.get(
            usage.user_tier,
            1.0
        )

        final_cost = round(base_cost * multiplier, 6)

        return UsageResult(
            total_tokens=total_tokens,
            base_cost=round(base_cost, 6),
            tier_multiplier=multiplier,
            final_cost=final_cost
        )

    # =====================================================
    # Budget Check
    # =====================================================

    def exceeds_budget(
        self,
        current_spend: float,
        new_cost: float,
        max_budget: float
    ) -> bool:

        return (current_spend + new_cost) > max_budget