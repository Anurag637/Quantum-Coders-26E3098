from dataclasses import dataclass
from typing import List, Dict
import re


# =========================================================
# Routing Result
# =========================================================

@dataclass
class RoutingDecision:
    task_type: str
    complexity: str
    requires_code: bool
    estimated_tokens: int
    suggested_tier: str


# =========================================================
# Prompt Analyzer
# =========================================================

class PromptAnalyzer:
    """
    Analyzes incoming prompt and suggests routing strategy.
    Pure domain logic.
    """

    CODE_PATTERNS = [
        r"\bdef\b",
        r"\bclass\b",
        r"\bimport\b",
        r"\bfunction\b",
        r"\bconsole\.log\b",
        r"\bSELECT\b",
        r"\bFROM\b",
        r"\bJOIN\b",
        r"\{.*\}",
    ]

    MATH_PATTERNS = [
        r"\bsolve\b",
        r"\bintegral\b",
        r"\bderivative\b",
        r"\bequation\b",
        r"\bmatrix\b",
        r"\bprobability\b",
    ]

    LONG_FORM_THRESHOLD = 400
    COMPLEXITY_TOKEN_THRESHOLD = 800

    # =====================================================
    # Main Analyze
    # =====================================================

    def analyze(self, prompt: str) -> RoutingDecision:

        prompt_lower = prompt.lower()
        token_estimate = self._estimate_tokens(prompt)

        requires_code = self._contains_pattern(
            prompt,
            self.CODE_PATTERNS
        )

        is_math = self._contains_pattern(
            prompt,
            self.MATH_PATTERNS
        )

        task_type = self._detect_task_type(
            requires_code=requires_code,
            is_math=is_math
        )

        complexity = self._estimate_complexity(
            prompt_length=len(prompt),
            token_estimate=token_estimate
        )

        suggested_tier = self._suggest_tier(
            requires_code=requires_code,
            complexity=complexity,
            token_estimate=token_estimate
        )

        return RoutingDecision(
            task_type=task_type,
            complexity=complexity,
            requires_code=requires_code,
            estimated_tokens=token_estimate,
            suggested_tier=suggested_tier,
        )

    # =====================================================
    # Helpers
    # =====================================================

    def _contains_pattern(self, text: str, patterns: List[str]) -> bool:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _estimate_tokens(self, text: str) -> int:
        # Simple heuristic: 1 token ≈ 4 chars
        return max(1, len(text) // 4)

    def _detect_task_type(
        self,
        requires_code: bool,
        is_math: bool,
    ) -> str:

        if requires_code:
            return "code"

        if is_math:
            return "math"

        return "general"

    def _estimate_complexity(
        self,
        prompt_length: int,
        token_estimate: int,
    ) -> str:

        if token_estimate > self.COMPLEXITY_TOKEN_THRESHOLD:
            return "high"

        if prompt_length > self.LONG_FORM_THRESHOLD:
            return "medium"

        return "low"

    def _suggest_tier(
        self,
        requires_code: bool,
        complexity: str,
        token_estimate: int,
    ) -> str:

        if requires_code:
            return "coder"

        if complexity == "high":
            return "large"

        if complexity == "medium":
            return "medium"

        return "small"