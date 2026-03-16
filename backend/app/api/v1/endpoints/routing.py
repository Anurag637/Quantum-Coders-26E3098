from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_db_session, require_api_key
from backend.app.schemas.routing import RoutingAnalyzeRequest, RoutingAnalyzeResponse
from backend.app.services.routing_service import RoutingService


router = APIRouter()


@router.post("/analyze", response_model=RoutingAnalyzeResponse)
async def analyze_routing(
    request: RoutingAnalyzeRequest,
    db: AsyncSession = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
):
    """
    Analyze a prompt and return:
    - Selected model
    - Alternative candidates
    - Strategy used
    - Estimated cost and latency
    """

    service = RoutingService(db)

    result = await service.analyze_prompt(request)

    return result