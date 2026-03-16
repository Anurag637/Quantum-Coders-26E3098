import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.models: dict[str, dict] = {}

    async def load_from_config(self, model_dir: str):
        """Populate the registry with supported models and their metadata."""
        logger.info("Loading models into registry...")
        self.models = {
            "google/gemma-7b-it": {
                "status": "available",
                "tier": "fast",
                "description": "Google Gemma 7B Instruct — fast, efficient model for simple tasks.",
                "max_tokens": 2048,
                "cost_per_1k_tokens": 0.0001,
            },
            "mistralai/Mistral-7B-Instruct-v0.2": {
                "status": "available",
                "tier": "capable",
                "description": "Mistral 7B Instruct v0.2 — high-quality reasoning and coding model.",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.0002,
            },
            "meta-llama/Llama-2-7b-chat-hf": {
                "status": "available",
                "tier": "standard",
                "description": "Meta LLaMA 2 7B Chat — versatile conversational model.",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.00015,
            },
        }
        logger.info("Loaded %d models into registry.", len(self.models))

    async def shutdown_all(self):
        """Gracefully clear the registry."""
        logger.info("Shutting down all models in registry...")
        self.models.clear()


model_registry = ModelRegistry()
