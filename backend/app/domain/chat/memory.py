from typing import List, Dict, Optional

from backend.app.domain.chat.conversation import Message


# =========================================================
# Domain Memory Policy
# =========================================================


class MemoryPolicy:
    """
    Defines how conversation memory is built
    before sending to model.
    """

    def __init__(
        self,
        max_tokens: int,
        tokenizer,
        system_prompt: Optional[str] = None,
    ):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    # =====================================================
    # Build Model Context
    # =====================================================

    def build_context(
        self,
        messages: List[Message],
    ) -> List[Dict[str, str]]:

        context: List[Dict[str, str]] = []
        total_tokens = 0

        # Add system prompt if exists
        if self.system_prompt:
            system_tokens = self.tokenizer(self.system_prompt)
            context.append(
                {"role": "system", "content": self.system_prompt}
            )
            total_tokens += system_tokens

        # Traverse newest → oldest
        for message in reversed(messages):
            tokens = self.tokenizer(message.content)

            if total_tokens + tokens > self.max_tokens:
                break

            context.insert(
                1 if self.system_prompt else 0,
                {
                    "role": message.role,
                    "content": message.content,
                },
            )

            total_tokens += tokens

        return context

    # =====================================================
    # Context Token Count
    # =====================================================

    def count_tokens(self, context: List[Dict[str, str]]) -> int:
        total = 0
        for msg in context:
            total += self.tokenizer(msg["content"])
        return total