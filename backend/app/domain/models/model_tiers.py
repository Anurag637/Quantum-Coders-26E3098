from enum import Enum
from typing import Optional


# =========================================================
# Tier Enum
# =========================================================

class ModelTier(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    CODER = "coder"


# =========================================================
# Tier Utilities
# =========================================================

class TierManager:
    """
    Central logic for tier ordering and transitions.
    """

    _ORDER = {
        ModelTier.SMALL: 1,
        ModelTier.MEDIUM: 2,
        ModelTier.LARGE: 3,
        ModelTier.CODER: 3,  # parallel to LARGE
    }

    # =====================================================
    # Comparison
    # =====================================================

    @classmethod
    def is_higher_or_equal(
        cls,
        tier_a: ModelTier,
        tier_b: ModelTier,
    ) -> bool:
        return cls._ORDER[tier_a] >= cls._ORDER[tier_b]

    # =====================================================
    # Upgrade
    # =====================================================

    @classmethod
    def upgrade(cls, tier: ModelTier) -> Optional[ModelTier]:

        if tier == ModelTier.SMALL:
            return ModelTier.MEDIUM

        if tier == ModelTier.MEDIUM:
            return ModelTier.LARGE

        return None

    # =====================================================
    # Downgrade
    # =====================================================

    @classmethod
    def downgrade(cls, tier: ModelTier) -> Optional[ModelTier]:

        if tier == ModelTier.LARGE:
            return ModelTier.MEDIUM

        if tier == ModelTier.MEDIUM:
            return ModelTier.SMALL

        if tier == ModelTier.CODER:
            return ModelTier.MEDIUM

        return None

    # =====================================================
    # Safe Tier Enforcement
    # =====================================================

    @classmethod
    def enforce_max_allowed(
        cls,
        requested: ModelTier,
        max_allowed: ModelTier,
    ) -> ModelTier:

        if cls.is_higher_or_equal(requested, max_allowed):
            return max_allowed

        return requested

    # =====================================================
    # Parse Tier
    # =====================================================

    @classmethod
    def from_string(cls, value: str) -> ModelTier:
        return ModelTier(value.lower())