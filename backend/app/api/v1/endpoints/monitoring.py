from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.services.monitoring_service import MonitoringService
from backend.app.core.security import get_api_key
from backend.app.persistence.session import get_db

router = APIRouter()

@router.get("/metrics")
async def get_metrics(
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    service = MonitoringService(db)
    return await service.get_system_metrics()
