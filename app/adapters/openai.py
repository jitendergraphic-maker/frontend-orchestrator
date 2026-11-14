from __future__ import annotations

import time

import httpx

from app.adapters.base import AdapterError, AdapterResult, ModelAdapter, normalize_messages
from app.config import get_settings


class OpenAIAdapter(ModelAdapter):
    provider = "openai"

    def complete(
        self,
        *,
        api_model: str,
        prompt: str,
        timeout: float,
        messages: list | None = None,
    ) -> AdapterResult:
        key = get_settings().openai_api_key
        if not key:
            raise AdapterError("OPENAI_API_KEY is not set")
        chat = normalize_messages(prompt=prompt, messages=messages)
        prompt_chars = sum(len(m["content"]) for m in chat)
        started = time.perf_counter()
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
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
            raise AdapterError("openai timeout") from exc
        except httpx.HTTPError as exc:
            raise AdapterError(f"openai transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise AdapterError(f"openai {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        choice = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AdapterResult(
            output=choice,
            tokens_in=int(usage.get("prompt_tokens") or max(8, prompt_chars // 4)),
            tokens_out=int(usage.get("completion_tokens") or max(8, len(choice) // 4)),
            latency_ms=latency_ms,
        )
