import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.models: dict[str, dict] = {}

    async def load_from_config(self, model_dir: str):
        """Populate the registry with supported models and their metadata."""
        logger.info("Loading models into registry...")
        # NOTE: IDs here are the exact model identifiers used by the underlying providers.
        # For Hugging Face models we keep the repo id; for Groq we use the Groq model name.
        self.models = {
            # ─── GROQ LIVE MODELS (verified via API key) ────────────────────
            "groq/llama-3.1-8b-instant": {
                "status": "available",
                "tier": "fast",
                "provider": "groq",
                "description": "Llama 3.1 8B Instant on Groq — ultra-fast, low-latency, general-purpose. ⚡ LIVE",
                "max_tokens": 131072,
                "cost_per_1k_tokens": 0.00005,
            },
            "groq/llama-3.3-70b-versatile": {
                "status": "available",
                "tier": "capable",
                "provider": "groq",
                "description": "Llama 3.3 70B Versatile on Groq — powerful, high-quality reasoning at speed. 🧠 LIVE",
                "max_tokens": 131072,
                "cost_per_1k_tokens": 0.00059,
            },
            "groq/meta-llama/llama-4-scout-17b-16e-instruct": {
                "status": "available",
                "tier": "premium",
                "provider": "groq",
                "description": "Llama 4 Scout 17B (16 Expert MoE) — Meta's newest multimodal model. 💎 LIVE",
                "max_tokens": 131072,
                "cost_per_1k_tokens": 0.00011,
            },
            "groq/moonshotai/kimi-k2-instruct": {
                "status": "available",
                "tier": "premium",
                "provider": "groq",
                "description": "Kimi K2 Instruct by Moonshot AI — strong reasoning & code model. 💎 LIVE",
                "max_tokens": 131072,
                "cost_per_1k_tokens": 0.0006,
            },
            "groq/qwen/qwen3-32b": {
                "status": "available",
                "tier": "capable",
                "provider": "groq",
                "description": "Qwen3 32B by Alibaba on Groq — multilingual, strong at instruction following. 🧠 LIVE",
                "max_tokens": 131072,
                "cost_per_1k_tokens": 0.00029,
            },
            "groq/compound": {
                "status": "available",
                "tier": "capable",
                "provider": "groq",
                "description": "Groq Compound — Groq's internal compound model for advanced tasks. 🧠 LIVE",
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.00045,
            },
            "groq/compound-mini": {
                "status": "available",
                "tier": "fast",
                "provider": "groq",
                "description": "Groq Compound Mini — faster, lightweight version of Compound. ⚡ LIVE",
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.00012,
            },
            # ─── HUGGING FACE COMMUNITY MODELS ──────────────────────────────
            "mistralai/Mistral-7B-Instruct-v0.2": {
                "status": "available",
                "tier": "capable",
                "provider": "huggingface",
                "description": "Mistral 7B Instruct v0.2 — high-quality reasoning and coding model.",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.00025,
            },
            "HuggingFaceH4/zephyr-7b-beta": {
                "status": "available",
                "tier": "capable",
                "provider": "huggingface",
                "description": "Zephyr 7B (Mistral-based instruction-tuned model).",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.00022,
            },
            "microsoft/phi-2": {
                "status": "available",
                "tier": "fast",
                "provider": "huggingface",
                "description": "Phi-2 (2.7B) — small, efficient model from Microsoft.",
                "max_tokens": 2048,
                "cost_per_1k_tokens": 0.00008,
            },
            "bigcode/starcoder2-7b": {
                "status": "available",
                "tier": "standard",
                "provider": "huggingface",
                "description": "StarCoder2 7B — improved open-source code generation model.",
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.00027,
            },
            "deepseek-ai/deepseek-coder-6.7b-instruct": {
                "status": "available",
                "tier": "capable",
                "provider": "huggingface",
                "description": "DeepSeek-Coder 6.7B — specialised code-focused instruction model.",
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.00028,
            },
            "Qwen/Qwen2-7B-Instruct": {
                "status": "available",
                "tier": "capable",
                "provider": "huggingface",
                "description": "Qwen2 7B Instruct — multilingual model with 32K context.",
                "max_tokens": 32768,
                "cost_per_1k_tokens": 0.00025,
            },
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
                "status": "available",
                "tier": "fast",
                "provider": "huggingface",
                "description": "TinyLlama 1.1B Chat — extremely lightweight, lowest latency.",
                "max_tokens": 2048,
                "cost_per_1k_tokens": 0.00003,
            },
        }
        logger.info("Loaded %d models into registry.", len(self.models))

    async def shutdown_all(self):
        """Gracefully clear the registry."""
        logger.info("Shutting down all models in registry...")
        self.models.clear()


model_registry = ModelRegistry()
