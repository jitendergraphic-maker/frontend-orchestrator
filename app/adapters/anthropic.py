from __future__ import annotations

import time

import httpx

from app.adapters.base import AdapterError, AdapterResult, ModelAdapter, normalize_messages
from app.config import get_settings


class AnthropicAdapter(ModelAdapter):
    provider = "anthropic"

    def complete(
        self,
        *,
        api_model: str,
        prompt: str,
        timeout: float,
        messages: list | None = None,
    ) -> AdapterResult:
        key = get_settings().anthropic_api_key
        if not key:
            raise AdapterError("ANTHROPIC_API_KEY is not set")
        chat = normalize_messages(prompt=prompt, messages=messages)
        system_parts = [m["content"] for m in chat if m["role"] == "system"]
        anth_messages = [m for m in chat if m["role"] in ("user", "assistant")]
        if not anth_messages:
            anth_messages = [{"role": "user", "content": prompt}]
        prompt_chars = sum(len(m["content"]) for m in chat)
        started = time.perf_counter()
        payload: dict = {
            "model": api_model,
            "max_tokens": get_settings().completion_max_tokens,
            "messages": anth_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise AdapterError("anthropic timeout") from exc
        except httpx.HTTPError as original:
            raise AdapterError(f"anthropic transport error: {original}") from original
        if resp.status_code >= 400:
            raise AdapterError(f"anthropic {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        parts = data.get("content") or []
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        usage = data.get("usage") or {}
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AdapterResult(
            output=text,
            tokens_in=int(usage.get("input_tokens") or max(8, prompt_chars // 4)),
            tokens_out=int(usage.get("output_tokens") or max(8, len(text) // 4)),
            latency_ms=latency_ms,
        )
