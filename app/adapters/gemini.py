from __future__ import annotations

import time

import httpx

from app.adapters.base import AdapterError, AdapterResult, ModelAdapter, normalize_messages
from app.config import get_settings


class GeminiAdapter(ModelAdapter):
    """Google Gemini via the OpenAI-compatible Generative Language API."""

    provider = "gemini"

    def complete(
        self,
        *,
        api_model: str,
        prompt: str,
        timeout: float,
        messages: list | None = None,
    ) -> AdapterResult:
        key = get_settings().gemini_api_key
        if not key:
            raise AdapterError("GEMINI_API_KEY is not set")

        chat = normalize_messages(prompt=prompt, messages=messages)
        prompt_chars = sum(len(m["content"]) for m in chat)
        started = time.perf_counter()
        response = None
        for attempt in range(2):
            try:
                response = httpx.post(
                    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": api_model,
                        "messages": chat,
                        "max_tokens": get_settings().completion_max_tokens,
                        "temperature": 0.2,
                    },
                    timeout=timeout,
                )
            except httpx.TimeoutException as exc:
                raise AdapterError("gemini timeout") from exc
            except httpx.HTTPError as exc:
                raise AdapterError(f"gemini transport error: {exc}") from exc

            if response.status_code != 429 or attempt == 1:
                break
            time.sleep(2)

        if response is None:
            raise AdapterError("gemini request failed")
        if response.status_code >= 400:
            raise AdapterError(
                f"gemini {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        output = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        return AdapterResult(
            output=output,
            tokens_in=int(
                usage.get("prompt_tokens") or max(8, prompt_chars // 4)
            ),
            tokens_out=int(
                usage.get("completion_tokens") or max(8, len(output) // 4)
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
