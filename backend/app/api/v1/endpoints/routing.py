from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.schemas.routing import RoutingAnalyzeRequest, RoutingAnalyzeResponse
from backend.app.services.routing_service import RoutingService
from backend.app.core.security import get_api_key
from backend.app.persistence.session import get_db

router = APIRouter()

@router.post("/analyze", response_model=RoutingAnalyzeResponse)
async def analyze_prompt(
    request: RoutingAnalyzeRequest,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    service = RoutingService(db)
    return await service.analyze_prompt(request)
