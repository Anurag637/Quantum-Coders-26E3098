from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_db_session, require_api_key
from backend.app.services.model_service import ModelService
from backend.app.schemas.model import ModelInfoResponse


router = APIRouter()


@router.get("", response_model=list[ModelInfoResponse])
async def list_models(
    db: AsyncSession = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
):
    """
    Returns all available models registered in the system.
    """
    service = ModelService(db)
    models = await service.list_models()
    return models


@router.get("/{model_id}", response_model=ModelInfoResponse)
async def get_model(
    model_id: str,
    db: AsyncSession = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
):
    """
    Returns details of a specific model.
    """
    service = ModelService(db)
    model = await service.get_model(model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return model