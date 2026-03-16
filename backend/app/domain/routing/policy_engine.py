from dataclasses import dataclass
from typing import List, Optional


# =========================================================
# Policy Context
# =========================================================

@dataclass
class PolicyContext:
    user_id: str
    user_tier: str                 # free | pro | enterprise
    estimated_tokens: int
    monthly_spend: float
    max_budget: float
    requested_model: Optional[str] = None


# =========================================================
# Policy Result
# =========================================================

@dataclass
class PolicyResult:
    allowed: bool
    reason: Optional[str] = None
    enforced_tier: Optional[str] = None
    token_limit: Optional[int] = None


# =========================================================
# Policy Engine
# =========================================================

class PolicyEngine:
    """
    Enforces system rules before final model selection.
    """

    FREE_TIER_TOKEN_LIMIT = 2000
    PRO_TIER_TOKEN_LIMIT = 8000
    ENTERPRISE_TOKEN_LIMIT = 32000

    # =====================================================
    # Main Policy Evaluation
    # =====================================================

    def evaluate(self, context: PolicyContext) -> PolicyResult:

        # -----------------------------
        # Budget enforcement
        # -----------------------------
        if context.monthly_spend >= context.max_budget:
            return PolicyResult(
                allowed=False,
                reason="Budget exceeded"
            )

        # -----------------------------
        # Tier restrictions
        # -----------------------------
        if context.user_tier == "free":
            return self._free_tier_policy(context)

        if context.user_tier == "pro":
            return self._pro_tier_policy(context)

        if context.user_tier == "enterprise":
            return self._enterprise_policy(context)

        return PolicyResult(
            allowed=False,
            reason="Invalid user tier"
        )

    # =====================================================
    # Tier Policies
    # =====================================================

    def _free_tier_policy(self, context: PolicyContext) -> PolicyResult:

        if context.estimated_tokens > self.FREE_TIER_TOKEN_LIMIT:
            return PolicyResult(
                allowed=False,
                reason="Token limit exceeded for free tier"
            )

        return PolicyResult(
            allowed=True,
            enforced_tier="small",
            token_limit=self.FREE_TIER_TOKEN_LIMIT
        )

    def _pro_tier_policy(self, context: PolicyContext) -> PolicyResult:

        if context.estimated_tokens > self.PRO_TIER_TOKEN_LIMIT:
            return PolicyResult(
                allowed=False,
                reason="Token limit exceeded for pro tier"
            )

        return PolicyResult(
            allowed=True,
            enforced_tier="medium",
            token_limit=self.PRO_TIER_TOKEN_LIMIT
        )

    def _enterprise_policy(self, context: PolicyContext) -> PolicyResult:

        if context.estimated_tokens > self.ENTERPRISE_TOKEN_LIMIT:
            return PolicyResult(
                allowed=False,
                reason="Token limit exceeded for enterprise tier"
            )

        return PolicyResult(
            allowed=True,
            enforced_tier="large",
            token_limit=self.ENTERPRISE_TOKEN_LIMIT
        )