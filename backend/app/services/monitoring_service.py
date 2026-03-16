from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional

from backend.app.persistence.models import InferenceLog


class MonitoringService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_system_metrics(self) -> dict:
        """Returns aggregated runtime metrics from the inference log table."""
        result = await self.db.execute(select(InferenceLog))
        logs = result.scalars().all()

        if not logs:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "cache_hit_rate": 0.0,
                "per_model_stats": {},
            }

        total = len(logs)
        success_count = sum(1 for log in logs if log.status == "success")
        error_count = total - success_count
        cache_hits = sum(1 for log in logs if log.cache_hit)
        avg_latency = sum(log.latency_ms or 0.0 for log in logs) / total

        per_model: dict[str, dict] = {}
        for log in logs:
            m = log.selected_model or "unknown"
            if m not in per_model:
                per_model[m] = {"requests": 0, "errors": 0, "total_latency": 0.0}
            per_model[m]["requests"] += 1
            if log.status != "success":
                per_model[m]["errors"] += 1
            per_model[m]["total_latency"] += log.latency_ms or 0.0

        per_model_stats = {
            m: {
                "requests": d["requests"],
                "error_rate": round(d["errors"] / d["requests"], 4) if d["requests"] else 0.0,
                "avg_latency_ms": round(d["total_latency"] / d["requests"], 2) if d["requests"] else 0.0,
            }
            for m, d in per_model.items()
        }

        return {
            "total_requests": total,
            "success_rate": round(success_count / total, 4),
            "error_rate": round(error_count / total, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "cache_hit_rate": round(cache_hits / total, 4),
            "per_model_stats": per_model_stats,
        }
