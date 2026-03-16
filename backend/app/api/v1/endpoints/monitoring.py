from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_db_session, require_api_key
from backend.app.services.monitoring_service import MonitoringService


router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic service health endpoint.
    Used by load balancers and Kubernetes probes.
    """
    return {
        "status": "healthy",
        "service": "llm-inference-gateway",
    }


@router.get("/metrics")
async def get_metrics(
    db: AsyncSession = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
):
    """
    Returns aggregated runtime metrics:
    - Total requests
    - Success rate
    - Error rate
    - Average latency
    - Per-model stats
    """

    service = MonitoringService(db)
    metrics = await service.get_system_metrics()

    return metrics