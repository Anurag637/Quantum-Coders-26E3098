from dataclasses import dataclass
from typing import Dict, List


# =========================================================
# Model Capability Profile
# =========================================================

@dataclass
class ModelCapability:
    name: str
    tier: str
    max_tokens: int
    supports_code: bool
    supports_creative: bool
    supports_reasoning: bool
    supports_chat: bool
    cost_per_1k_tokens: float


# =========================================================
# Capability Matrix
# =========================================================

class CapabilityMatrix:
    """
    Central capability definition for all models.
    This is the single source of truth.
    """

    def __init__(self):
        self._models: Dict[str, ModelCapability] = {}
        self._load_defaults()

    # =====================================================
    # Default Model Definitions
    # =====================================================

    def _load_defaults(self):

        self._register(ModelCapability(
            name="llama-3.1-8b-instant",
            tier="large",
            max_tokens=8192,
            supports_code=True,
            supports_creative=True,
            supports_reasoning=True,
            supports_chat=True,
            cost_per_1k_tokens=0.006
        ))

        self._register(ModelCapability(
            name="Mistral 7B",
            tier="medium",
            max_tokens=8192,
            supports_code=False,
            supports_creative=False,
            supports_reasoning=True,
            supports_chat=True,
            cost_per_1k_tokens=0.004
        ))

        self._register(ModelCapability(
            name="Mistral 7B Instruct",
            tier="medium",
            max_tokens=8192,
            supports_code=True,
            supports_creative=True,
            supports_reasoning=True,
            supports_chat=True,
            cost_per_1k_tokens=0.0045
        ))

        self._register(ModelCapability(
            name="DeepSeek-Coder 6.7B",
            tier="coder",
            max_tokens=16384,
            supports_code=True,
            supports_creative=False,
            supports_reasoning=True,
            supports_chat=True,
            cost_per_1k_tokens=0.005
        ))

        self._register(ModelCapability(
            name="StarCoder2 7B",
            tier="coder",
            max_tokens=16384,
            supports_code=True,
            supports_creative=False,
            supports_reasoning=False,
            supports_chat=False,
            cost_per_1k_tokens=0.0055
        ))

        self._register(ModelCapability(
            name="TinyLlama 1.1B",
            tier="small",
            max_tokens=2048,
            supports_code=False,
            supports_creative=False,
            supports_reasoning=False,
            supports_chat=True,
            cost_per_1k_tokens=0.001
        ))

        self._register(ModelCapability(
            name="Phi-2",
            tier="small",
            max_tokens=2048,
            supports_code=True,
            supports_creative=False,
            supports_reasoning=True,
            supports_chat=True,
            cost_per_1k_tokens=0.0015
        ))

        self._register(ModelCapability(
            name="Qwen2 7B",
            tier="medium",
            max_tokens=8192,
            supports_code=True,
            supports_creative=True,
            supports_reasoning=True,
            supports_chat=True,
            cost_per_1k_tokens=0.004
        ))

    # =====================================================
    # Registration
    # =====================================================

    def _register(self, model: ModelCapability):
        self._models[model.name] = model

    # =====================================================
    # Public API
    # =====================================================

    def get(self, name: str) -> ModelCapability:
        return self._models[name]

    def all_models(self) -> List[ModelCapability]:
        return list(self._models.values())

    def by_tier(self, tier: str) -> List[ModelCapability]:
        return [m for m in self._models.values() if m.tier == tier]

    def capable_of_code(self) -> List[ModelCapability]:
        return [m for m in self._models.values() if m.supports_code]

    def capable_of_creative(self) -> List[ModelCapability]:
        return [m for m in self._models.values() if m.supports_creative]