from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import uuid


# =========================================================
# Domain Message
# =========================================================

@dataclass
class Message:
    role: str
    content: str
    model_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def is_user(self) -> bool:
        return self.role == "user"

    def is_assistant(self) -> bool:
        return self.role == "assistant"

    def is_system(self) -> bool:
        return self.role == "system"


# =========================================================
# Domain Conversation
# =========================================================

@dataclass
class Conversation:
    user_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # =====================================================
    # Behavior Methods
    # =====================================================

    def add_message(
        self,
        role: str,
        content: str,
        model_name: Optional[str] = None,
    ):
        message = Message(
            role=role,
            content=content,
            model_name=model_name,
        )
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def get_last_user_message(self) -> Optional[Message]:
        for message in reversed(self.messages):
            if message.is_user():
                return message
        return None

    def message_count(self) -> int:
        return len(self.messages)

    def has_messages(self) -> bool:
        return len(self.messages) > 0