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
            # Groq-hosted
            "groq/llama-3.1-8b-instant": {
                "status": "available",
                "tier": "fast",
                "provider": "groq",
                "description": "Llama 3.1 8B Instant on Groq — very fast general-purpose model.",
                "max_tokens": 131072,
                "cost_per_1k_tokens": 0.00005,
            },
            # Hugging Face hosted / community models
            "mistralai/Mistral-7B-v0.1": {
                "status": "available",
                "tier": "standard",
                "provider": "huggingface",
                "description": "Base Mistral 7B model.",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.0002,
            },
            "mistralai/Mistral-7B-Instruct-v0.2": {
                "status": "available",
                "tier": "capable",
                "provider": "huggingface",
                "description": "Mistral 7B Instruct v0.2 — high-quality reasoning and coding model.",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.00025,
            },
            "tiiuae/falcon-7b-instruct": {
                "status": "available",
                "tier": "standard",
                "provider": "huggingface",
                "description": "Falcon 7B Instruct.",
                "max_tokens": 2048,
                "cost_per_1k_tokens": 0.00018,
            },
            "openchat/openchat-3.5": {
                "status": "available",
                "tier": "capable",
                "provider": "huggingface",
                "description": "OpenChat 3.5 (Mistral-based conversational model).",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.0002,
            },
            "HuggingFaceH4/zephyr-7b-beta": {
                "status": "available",
                "tier": "capable",
                "provider": "huggingface",
                "description": "Zephyr 7B (Mistral-based instruction-tuned model).",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.00022,
            },
            "TheBloke/NeuralBeagle14-7B-GGUF": {
                "status": "experimental",
                "tier": "standard",
                "provider": "huggingface",
                "description": "NeuralBeagle 7B — experimental conversational model.",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.0002,
            },
            "bigcode/starcoderbase-7b": {
                "status": "available",
                "tier": "standard",
                "provider": "huggingface",
                "description": "StarCoderBase 7B — code generation model.",
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.00025,
            },
            "bigcode/starcoder2-7b": {
                "status": "available",
                "tier": "standard",
                "provider": "huggingface",
                "description": "StarCoder2 7B — improved code model.",
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.00027,
            },
            "microsoft/phi-2": {
                "status": "available",
                "tier": "fast",
                "provider": "huggingface",
                "description": "Phi-2 (2.7B) — small, efficient model.",
                "max_tokens": 2048,
                "cost_per_1k_tokens": 0.00008,
            },
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
                "status": "available",
                "tier": "fast",
                "provider": "huggingface",
                "description": "TinyLlama 1.1B Chat — very small, low-cost model.",
                "max_tokens": 2048,
                "cost_per_1k_tokens": 0.00003,
            },
            "EleutherAI/gpt-j-6B": {
                "status": "available",
                "tier": "standard",
                "provider": "huggingface",
                "description": "GPT-J 6B — general language model.",
                "max_tokens": 2048,
                "cost_per_1k_tokens": 0.0002,
            },
            "togethercomputer/RedPajama-INCITE-7B-Chat": {
                "status": "available",
                "tier": "standard",
                "provider": "huggingface",
                "description": "RedPajama-INCITE 7B Chat.",
                "max_tokens": 2048,
                "cost_per_1k_tokens": 0.00018,
            },
            "mosaicml/mpt-7b-chat": {
                "status": "available",
                "tier": "standard",
                "provider": "huggingface",
                "description": "MPT-7B Chat.",
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.00022,
            },
            "Qwen/Qwen2-7B-Instruct": {
                "status": "available",
                "tier": "capable",
                "provider": "huggingface",
                "description": "Qwen2 7B Instruct (can be deployed with quantization).",
                "max_tokens": 32768,
                "cost_per_1k_tokens": 0.00025,
            },
            "deepseek-ai/deepseek-coder-6.7b-instruct": {
                "status": "available",
                "tier": "capable",
                "provider": "huggingface",
                "description": "DeepSeek-Coder 6.7B — code-focused model.",
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.00028,
            },
        }
        logger.info("Loaded %d models into registry.", len(self.models))

    async def shutdown_all(self):
        """Gracefully clear the registry."""
        logger.info("Shutting down all models in registry...")
        self.models.clear()


model_registry = ModelRegistry()
