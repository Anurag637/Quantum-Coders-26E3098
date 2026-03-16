from pydantic import BaseModel
from typing import Optional


class ModelInfoResponse(BaseModel):
    id: str
    status: str
    tier: str
    description: Optional[str] = None
    max_tokens: Optional[int] = None
    cost_per_1k_tokens: Optional[float] = None
