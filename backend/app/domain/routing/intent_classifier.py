from dataclasses import dataclass
from typing import List
import re


# =========================================================
# Intent Result
# =========================================================

@dataclass
class IntentResult:
    intent: str
    confidence: float


# =========================================================
# Intent Classifier
# =========================================================

class IntentClassifier:
    """
    Classifies high-level user intent.
    Pure domain logic.
    No model calls.
    """

    CODE_PATTERNS = [
        r"\bwrite code\b",
        r"\bdebug\b",
        r"\bfix this code\b",
        r"\bpython\b",
        r"\bjavascript\b",
        r"\bjava\b",
        r"\bc\+\+\b",
        r"\bapi endpoint\b",
        r"\bsql query\b",
    ]

    EXPLANATION_PATTERNS = [
        r"\bexplain\b",
        r"\bwhat is\b",
        r"\bhow does\b",
        r"\bdescribe\b",
        r"\bwhy does\b",
    ]

    CREATIVE_PATTERNS = [
        r"\bstory\b",
        r"\bpoem\b",
        r"\bcreative\b",
        r"\bnovel\b",
        r"\bscript\b",
    ]

    SUMMARIZATION_PATTERNS = [
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\btl;dr\b",
    ]

    TRANSLATION_PATTERNS = [
        r"\btranslate\b",
        r"\bin spanish\b",
        r"\bin french\b",
        r"\bin hindi\b",
    ]

    # =====================================================
    # Main Method
    # =====================================================

    def classify(self, prompt: str) -> IntentResult:

        prompt_lower = prompt.lower()

        if self._matches(prompt_lower, self.CODE_PATTERNS):
            return IntentResult(intent="code_generation", confidence=0.9)

        if self._matches(prompt_lower, self.SUMMARIZATION_PATTERNS):
            return IntentResult(intent="summarization", confidence=0.85)

        if self._matches(prompt_lower, self.TRANSLATION_PATTERNS):
            return IntentResult(intent="translation", confidence=0.85)

        if self._matches(prompt_lower, self.CREATIVE_PATTERNS):
            return IntentResult(intent="creative_writing", confidence=0.8)

        if self._matches(prompt_lower, self.EXPLANATION_PATTERNS):
            return IntentResult(intent="explanation", confidence=0.75)

        return IntentResult(intent="general_chat", confidence=0.6)

    # =====================================================
    # Helper
    # =====================================================

    def _matches(self, text: str, patterns: List[str]) -> bool:
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False