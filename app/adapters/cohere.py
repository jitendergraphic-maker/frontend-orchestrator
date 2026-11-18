from __future__ import annotations

import time
from typing import Any

import httpx

from app.adapters.base import AdapterError, AdapterResult, ModelAdapter, normalize_messages
from app.config import get_settings


class CohereAdapter(ModelAdapter):
    """Cohere Chat API v2 (trial keys are free with monthly call limits)."""

    provider = "cohere"

    def complete(
        self,
        *,
        api_model: str,
        prompt: str,
        timeout: float,
        messages: list | None = None,
    ) -> AdapterResult:
        key = get_settings().cohere_api_key
        if not key:
            raise AdapterError("COHERE_API_KEY is not set")

        chat = normalize_messages(prompt=prompt, messages=messages)
        prompt_chars = sum(len(m["content"]) for m in chat)
        started = time.perf_counter()
        response = None
        for attempt in range(2):
            try:
                response = httpx.post(
                    "https://api.cohere.com/v2/chat",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json={
                        "model": api_model,
                        "messages": chat,
                        "max_tokens": get_settings().completion_max_tokens,
                        "temperature": 0.2,
                    },
                    timeout=timeout,
                )
            except httpx.TimeoutException as exc:
                raise AdapterError("cohere timeout") from exc
            except httpx.HTTPError as exc:
                raise AdapterError(f"cohere transport error: {exc}") from exc

            if response.status_code != 429 or attempt == 1:
                break
            time.sleep(2)

        if response is None:
            raise AdapterError("cohere request failed")
        if response.status_code >= 400:
            raise AdapterError(
                f"cohere {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        output = _extract_text(data)
        usage = data.get("usage") or {}
        billed = usage.get("billed_units") if isinstance(usage.get("billed_units"), dict) else {}
        tokens = usage.get("tokens") if isinstance(usage.get("tokens"), dict) else {}
        return AdapterResult(
            output=output,
            tokens_in=int(
                billed.get("input_tokens")
                or tokens.get("input_tokens")
                or max(8, prompt_chars // 4)
            ),
            tokens_out=int(
                billed.get("output_tokens")
                or tokens.get("output_tokens")
                or max(8, len(output) // 4)
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def _extract_text(data: Any) -> str:
    if not isinstance(data, dict):
        return str(data or "")
    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            parts = [
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            ]
            text = "".join(parts).strip()
            if text:
                return text
        if isinstance(content, str) and content.strip():
            return content
    if data.get("text"):
        return str(data["text"])
    return ""
