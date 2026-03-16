from dataclasses import dataclass
from typing import Dict


# =========================================================
# Cost Profile
# =========================================================

@dataclass
class CostProfile:
    model_name: str
    input_cost_per_1k: float
    output_cost_per_1k: float


# =========================================================
# Cost Manager
# =========================================================

class CostProfiles:
    """
    Centralized cost definitions.
    Used by scoring engine and billing.
    """

    def __init__(self):
        self._profiles: Dict[str, CostProfile] = {}
        self._load_defaults()

    # =====================================================
    # Default Model Cost Definitions
    # =====================================================

    def _load_defaults(self):

        self._register("TinyLlama 1.1B", 0.001, 0.001)
        self._register("Phi-2", 0.0015, 0.0015)

        self._register("Mistral 7B", 0.004, 0.004)
        self._register("Mistral 7B Instruct", 0.0045, 0.0045)
        self._register("Qwen2 7B", 0.004, 0.004)

        self._register("llama-3.1-8b-instant", 0.006, 0.006)

        self._register("DeepSeek-Coder 6.7B", 0.005, 0.005)
        self._register("StarCoder2 7B", 0.0055, 0.0055)

    def _register(self, name: str, input_cost: float, output_cost: float):
        self._profiles[name] = CostProfile(
            model_name=name,
            input_cost_per_1k=input_cost,
            output_cost_per_1k=output_cost
        )

    # =====================================================
    # Public API
    # =====================================================

    def estimate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:

        profile = self._profiles[model_name]

        input_cost = (input_tokens / 1000) * profile.input_cost_per_1k
        output_cost = (output_tokens / 1000) * profile.output_cost_per_1k

        return round(input_cost + output_cost, 6)

    def get(self, model_name: str) -> CostProfile:
        return self._profiles[model_name]