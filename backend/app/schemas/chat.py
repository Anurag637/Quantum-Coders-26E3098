from pydantic import BaseModel
from typing import Optional, List


class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


class ChatResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    model: str
    choices: List[ChatResponseChoice]
    usage: dict
