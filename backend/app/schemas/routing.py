from pydantic import BaseModel
from typing import Optional, List

class RoutingAnalyzeRequest(BaseModel):
    prompt: str
    preferred_model: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7

class RoutingAnalyzeResponse(BaseModel):
    selected_model: str
    alternative_candidates: List[str]
    strategy_used: str
    estimated_cost: float
    estimated_latency: float
    response_text: str = ""
