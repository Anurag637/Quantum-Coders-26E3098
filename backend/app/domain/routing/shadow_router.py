from dataclasses import dataclass
from typing import Optional, List


# =========================================================
# Shadow Configuration
# =========================================================

@dataclass
class ShadowConfig:
    enabled: bool
    traffic_percentage: float  # 0.0 - 100.0
    shadow_model_name: Optional[str] = None


# =========================================================
# Shadow Decision
# =========================================================

@dataclass
class ShadowDecision:
    use_shadow: bool
    primary_model: str
    shadow_model: Optional[str]
    reason: str


# =========================================================
# Shadow Router
# =========================================================

class ShadowRouter:
    """
    Determines whether a request should also be routed
    to a shadow model for evaluation.
    """

    def __init__(self, config: ShadowConfig):
        self.config = config

    # =====================================================
    # Decide Shadow Execution
    # =====================================================

    def evaluate(
        self,
        primary_model: str,
        user_id: str,
    ) -> ShadowDecision:

        if not self.config.enabled:
            return ShadowDecision(
                use_shadow=False,
                primary_model=primary_model,
                shadow_model=None,
                reason="Shadow routing disabled"
            )

        if not self.config.shadow_model_name:
            return ShadowDecision(
                use_shadow=False,
                primary_model=primary_model,
                shadow_model=None,
                reason="No shadow model configured"
            )

        if not self._should_sample(user_id):
            return ShadowDecision(
                use_shadow=False,
                primary_model=primary_model,
                shadow_model=None,
                reason="Traffic sampling skipped"
            )

        if primary_model == self.config.shadow_model_name:
            return ShadowDecision(
                use_shadow=False,
                primary_model=primary_model,
                shadow_model=None,
                reason="Primary equals shadow model"
            )

        return ShadowDecision(
            use_shadow=True,
            primary_model=primary_model,
            shadow_model=self.config.shadow_model_name,
            reason="Shadow routing enabled for sampled request"
        )

    # =====================================================
    # Deterministic Sampling
    # =====================================================

    def _should_sample(self, user_id: str) -> bool:
        """
        Deterministic percentage-based sampling.
        Ensures same user consistently sampled.
        """
        hash_value = abs(hash(user_id)) % 100
        return hash_value < self.config.traffic_percentage