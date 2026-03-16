from typing import Dict, Optional
from functools import lru_cache

from backend.app.core.config import get_settings


settings = get_settings()


class FeatureFlags:
    """
    Central feature flag manager.

    Supports:
    - Global flags (env-based)
    - Per-user flags
    - Per-model flags
    """

    def __init__(self):
        # Global feature flags from environment
        self._flags: Dict[str, bool] = {
            "enable_streaming": settings.ENABLE_STREAMING,
            "enable_metrics": settings.ENABLE_METRICS,
            "enable_tracing": settings.ENABLE_TRACING,
            "enable_semantic_cache": True,
            "enable_rate_limiting": True,
            "enable_circuit_breaker": True,
            "enable_billing": True,
            "enable_model_routing": True,
        }

        # Optional runtime overrides
        self._runtime_flags: Dict[str, bool] = {}

        # Optional per-user flags
        self._user_flags: Dict[str, Dict[str, bool]] = {}

        # Optional per-model flags
        self._model_flags: Dict[str, Dict[str, bool]] = {}

    # ===============================
    # Global Flags
    # ===============================

    def is_enabled(self, flag: str) -> bool:
        if flag in self._runtime_flags:
            return self._runtime_flags[flag]
        return self._flags.get(flag, False)

    def set_runtime_flag(self, flag: str, value: bool):
        self._runtime_flags[flag] = value

    # ===============================
    # User Flags
    # ===============================

    def enable_for_user(self, user_id: str, flag: str):
        self._user_flags.setdefault(user_id, {})[flag] = True

    def disable_for_user(self, user_id: str, flag: str):
        self._user_flags.setdefault(user_id, {})[flag] = False

    def is_enabled_for_user(self, user_id: str, flag: str) -> bool:
        if user_id in self._user_flags and flag in self._user_flags[user_id]:
            return self._user_flags[user_id][flag]
        return self.is_enabled(flag)

    # ===============================
    # Model Flags
    # ===============================

    def enable_for_model(self, model_name: str, flag: str):
        self._model_flags.setdefault(model_name, {})[flag] = True

    def disable_for_model(self, model_name: str, flag: str):
        self._model_flags.setdefault(model_name, {})[flag] = False

    def is_enabled_for_model(self, model_name: str, flag: str) -> bool:
        if model_name in self._model_flags and flag in self._model_flags[model_name]:
            return self._model_flags[model_name][flag]
        return self.is_enabled(flag)


# Singleton instance
@lru_cache
def get_feature_flags() -> FeatureFlags:
    return FeatureFlags()