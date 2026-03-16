from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime


class BillingSummaryResponse(BaseModel):
    total_requests: int
    total_cost: float
    total_tokens_estimated: int
    cost_by_model: Dict[str, float]
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class BillingDetailResponse(BaseModel):
    request_id: str
    timestamp: Optional[datetime]
    selected_model: str
    prompt_preview: str
    latency_ms: float
    cost_estimate: float
    status: str
    cache_hit: bool
