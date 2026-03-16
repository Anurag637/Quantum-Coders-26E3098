from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.models.registry import model_registry
from backend.app.schemas.model import ModelInfoResponse


class ModelService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_models(self) -> list[ModelInfoResponse]:
        return [
            ModelInfoResponse(
                id=model_id,
                status=data.get("status", "unknown"),
                tier=data.get("tier", "unknown"),
                description=data.get("description"),
                max_tokens=data.get("max_tokens"),
                cost_per_1k_tokens=data.get("cost_per_1k_tokens"),
            )
            for model_id, data in model_registry.models.items()
        ]

    async def get_model(self, model_id: str) -> ModelInfoResponse | None:
        data = model_registry.models.get(model_id)
        if not data:
            return None
        return ModelInfoResponse(
            id=model_id,
            status=data.get("status", "unknown"),
            tier=data.get("tier", "unknown"),
            description=data.get("description"),
            max_tokens=data.get("max_tokens"),
            cost_per_1k_tokens=data.get("cost_per_1k_tokens"),
        )
