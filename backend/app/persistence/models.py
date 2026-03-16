import os
from sqlalchemy import Column, String, Float, DateTime, Text, Integer, JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()

class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    request_id = Column(String, index=True)
    prompt = Column(Text, nullable=False)
    selected_model = Column(String, index=True)
    strategy_used = Column(String)
    response_text = Column(Text)
    latency_ms = Column(Float)
    cost_estimate = Column(Float)
    status = Column(String) # success, error
    error_message = Column(Text, nullable=True)
    cache_hit = Column(Integer, default=0) # bool integer
