import logging
from typing import Dict, Any, List

import httpx

from backend.app.core.config import get_settings


logger = logging.getLogger(__name__)


class GroqClient:
    """
    Minimal Groq API client for llama-3.1-8b-instant.
    Uses the HTTP REST API instead of the groq SDK to keep dependencies light.
    """

    def __init__(self) -> None:
        # Read from central settings (.env is already loaded there)
        settings = get_settings()
        self.api_key = getattr(settings, "GROQ_API_KEY", None)
        if self.api_key:
            self.api_key = self.api_key.strip()
        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY environment variable not set. "
                "Calls to llama-3.1-8b-instant will be mocked."
            )
        self.base_url = "https://api.groq.com/openai/v1"

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Call Groq's OpenAI-compatible chat completions endpoint.
        Returns a dict with keys: {text, status, error?}
        """
        # If no key is configured, return a mocked response so the rest of
        # the platform keeps working in local dev.
        if not self.api_key:
            import asyncio

            await asyncio.sleep(0.5)
            joined = "\\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return {
                "text": (
                    f"[Mock Groq / {model}]\\n\\n"
                    f"Messages:\\n{joined}\\n\\n"
                    "Set GROQ_API_KEY in .env to get real completions."
                ),
                "status": "success",
            }

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                return {"text": content, "status": "success"}
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Groq API HTTP error: %s - %s",
                    exc.response.status_code,
                    exc.response.text,
                )
                return {
                    "status": "error",
                    "error": f"API Error: {exc.response.status_code}",
                }
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Groq API connection error: %s", exc)
                return {"status": "error", "error": f"Connection Error: {exc}"}

