from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from datetime import datetime

from backend.app.api.deps import get_db_session, require_api_key, get_current_user
from backend.app.services.billing_service import BillingService
from backend.app.schemas.billing import (
    BillingSummaryResponse,
    BillingDetailResponse,
)


router = APIRouter()


@router.get("/summary", response_model=BillingSummaryResponse)
async def get_billing_summary(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
    user=Depends(get_current_user),
):
    """
    Returns billing summary for the authenticated user:
    - Total tokens used
    - Total cost
    - Requests count
    - Cost per model
    """

    service = BillingService(db)

    summary = await service.get_user_summary(
        user_id=user.id,
        start_date=start_date,
        end_date=end_date,
    )

    return summary


@router.get("/details", response_model=list[BillingDetailResponse])
async def get_billing_details(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    model: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
    user=Depends(get_current_user),
):
    """
    Returns detailed billing records:
    - Per request cost
    - Tokens used
    - Model used
    - Timestamp
    """

    service = BillingService(db)

    details = await service.get_user_details(
        user_id=user.id,
        start_date=start_date,
        end_date=end_date,
        model=model,
    )

    return details


@router.get("/admin/overview")
async def get_admin_billing_overview(
    db: AsyncSession = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
):
    """
    Admin endpoint:
    Returns system-wide billing statistics.
    """

    service = BillingService(db)

    overview = await service.get_system_overview()

    return overview