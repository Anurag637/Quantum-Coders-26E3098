from dataclasses import dataclass
from typing import Dict


# =========================================================
# Performance Profile
# =========================================================

@dataclass
class PerformanceProfile:
    model_name: str
    avg_latency_ms: float
    max_rps: float
    health_score: float  # 0.0 - 1.0
    current_load: float  # 0.0 - 1.0


# =========================================================
# Performance Manager
# =========================================================

class PerformanceProfiles:
    """
    Runtime performance intelligence.
    Updated dynamically from monitoring system.
    """

    def __init__(self):
        self._profiles: Dict[str, PerformanceProfile] = {}
        self._load_defaults()

    # =====================================================
    # Default Boot Profiles
    # =====================================================

    def _load_defaults(self):

        self._register("TinyLlama 1.1B", 80, 50)
        self._register("Phi-2", 90, 40)

        self._register("Mistral 7B", 150, 30)
        self._register("Mistral 7B Instruct", 170, 25)

        self._register("Qwen2 7B", 160, 28)

        self._register("llama-3.1-8b-instant", 250, 15)

        self._register("DeepSeek-Coder 6.7B", 220, 20)
        self._register("StarCoder2 7B", 240, 18)

    def _register(self, name: str, latency: float, rps: float):
        self._profiles[name] = PerformanceProfile(
            model_name=name,
            avg_latency_ms=latency,
            max_rps=rps,
            health_score=1.0,
            current_load=0.0
        )

    # =====================================================
    # Public API
    # =====================================================

    def update_metrics(
        self,
        model_name: str,
        latency_ms: float,
        current_load: float,
        health_score: float
    ):
        profile = self._profiles[model_name]
        profile.avg_latency_ms = latency_ms
        profile.current_load = current_load
        profile.health_score = health_score

    def get(self, model_name: str) -> PerformanceProfile:
        return self._profiles[model_name]

    def all(self):
        return list(self._profiles.values())