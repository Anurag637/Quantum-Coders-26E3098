from fastapi import APIRouter, Depends
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.schemas.model import ModelInfoResponse
from backend.app.services.model_service import ModelService
from backend.app.core.security import get_api_key
from backend.app.persistence.session import get_db

router = APIRouter()

@router.get("", response_model=List[ModelInfoResponse])
async def list_models(
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    service = ModelService(db)
    return await service.list_models()
