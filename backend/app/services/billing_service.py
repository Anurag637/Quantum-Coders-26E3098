from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
from datetime import datetime

from backend.app.persistence.models import InferenceLog
from backend.app.schemas.billing import BillingSummaryResponse, BillingDetailResponse


class BillingService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_summary(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BillingSummaryResponse:
        """Aggregate billing info from inference logs."""
        stmt = select(InferenceLog)
        if start_date:
            stmt = stmt.where(InferenceLog.timestamp >= start_date)
        if end_date:
            stmt = stmt.where(InferenceLog.timestamp <= end_date)

        result = await self.db.execute(stmt)
        logs = result.scalars().all()

        total_cost = sum(log.cost_estimate or 0.0 for log in logs)
        total_requests = len(logs)
        # Estimate tokens: ~4 chars per token on the prompt side
        total_tokens = sum(len(log.prompt or "") // 4 for log in logs)

        cost_by_model: dict[str, float] = {}
        for log in logs:
            model = log.selected_model or "unknown"
            cost_by_model[model] = cost_by_model.get(model, 0.0) + (log.cost_estimate or 0.0)

        return BillingSummaryResponse(
            total_requests=total_requests,
            total_cost=round(total_cost, 6),
            total_tokens_estimated=total_tokens,
            cost_by_model={k: round(v, 6) for k, v in cost_by_model.items()},
            period_start=start_date,
            period_end=end_date,
        )

    async def get_user_details(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        model: Optional[str] = None,
    ) -> list[BillingDetailResponse]:
        stmt = select(InferenceLog).order_by(InferenceLog.timestamp.desc()).limit(100)
        if start_date:
            stmt = stmt.where(InferenceLog.timestamp >= start_date)
        if end_date:
            stmt = stmt.where(InferenceLog.timestamp <= end_date)
        if model:
            stmt = stmt.where(InferenceLog.selected_model == model)

        result = await self.db.execute(stmt)
        logs = result.scalars().all()

        return [
            BillingDetailResponse(
                request_id=log.request_id,
                timestamp=log.timestamp,
                selected_model=log.selected_model or "unknown",
                prompt_preview=(log.prompt or "")[:80] + ("..." if len(log.prompt or "") > 80 else ""),
                latency_ms=log.latency_ms or 0.0,
                cost_estimate=log.cost_estimate or 0.0,
                status=log.status or "unknown",
                cache_hit=bool(log.cache_hit),
            )
            for log in logs
        ]

    async def get_system_overview(self) -> dict:
        result = await self.db.execute(select(InferenceLog))
        logs = result.scalars().all()
        total_cost = sum(log.cost_estimate or 0.0 for log in logs)
        total_requests = len(logs)
        success_count = sum(1 for log in logs if log.status == "success")
        return {
            "total_requests": total_requests,
            "total_cost": round(total_cost, 6),
            "success_rate": round(success_count / total_requests, 4) if total_requests else 0.0,
            "unique_models": list({log.selected_model for log in logs if log.selected_model}),
        }
