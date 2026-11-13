from __future__ import annotations

import time

import httpx

from app.adapters.base import AdapterError, AdapterResult, ModelAdapter
from app.config import get_settings


class OpenRouterAdapter(ModelAdapter):
    """OpenAI-compatible client for OpenRouter (many models, one API key)."""

    provider = "openrouter"

    def complete(
        self, *, api_model: str, prompt: str, timeout: float
    ) -> AdapterResult:
        key = get_settings().openrouter_api_key
        if not key:
            raise AdapterError("OPENROUTER_API_KEY is not set")

        started = time.perf_counter()
        response = None
        for attempt in range(2):
            try:
                response = httpx.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "HTTP-Referer": "http://localhost:5173",
                        "X-Title": "Orchestra LLM Orchestration Engine",
                    },
                    json={
                        "model": api_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": get_settings().completion_max_tokens,
                        "temperature": 0.2,
                    },
                    timeout=timeout,
                )
            except httpx.TimeoutException as exc:
                raise AdapterError("openrouter timeout") from exc
            except httpx.HTTPError as exc:
                raise AdapterError(f"openrouter transport error: {exc}") from exc

            if response.status_code != 429 or attempt == 1:
                break
            time.sleep(2)

        if response is None:
            raise AdapterError("openrouter request failed")
        if response.status_code >= 400:
            raise AdapterError(
                f"openrouter {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        output = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        return AdapterResult(
            output=output,
            tokens_in=int(
                usage.get("prompt_tokens") or max(8, len(prompt) // 4)
            ),
            tokens_out=int(
                usage.get("completion_tokens") or max(8, len(output) // 4)
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
