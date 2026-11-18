from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import httpx

from app.adapters.base import AdapterError, AdapterResult, ModelAdapter, normalize_messages
from app.config import get_settings


class CloudflareAdapter(ModelAdapter):
    provider = "cloudflare"

    def complete(
        self,
        *,
        api_model: str,
        prompt: str,
        timeout: float,
        messages: list | None = None,
    ) -> AdapterResult:
        settings = get_settings()
        token = settings.cloudflare_api_token
        account_id = settings.cloudflare_account_id
        if not token or not account_id:
            raise AdapterError("CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID is not set")

        chat = normalize_messages(prompt=prompt, messages=messages)
        prompt_chars = sum(len(m["content"]) for m in chat)
        # Model ids contain "@" and "/"; encode path segments but keep slashes.
        model_path = quote(api_model, safe="/@.-_")
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
            f"/ai/run/{model_path}"
        )
        started = time.perf_counter()
        try:
            response = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "messages": chat,
                    "max_tokens": settings.completion_max_tokens,
                },
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise AdapterError("cloudflare timeout") from exc
        except httpx.HTTPError as exc:
            raise AdapterError(f"cloudflare transport error: {exc}") from exc

        if response.status_code >= 400:
            raise AdapterError(
                f"cloudflare {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        output, usage = _extract_output(data)
        return AdapterResult(
            output=output,
            tokens_in=int(
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or max(8, prompt_chars // 4)
            ),
            tokens_out=int(
                usage.get("completion_tokens")
                or usage.get("output_tokens")
                or max(8, len(output) // 4)
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def _extract_output(data: Any) -> tuple[str, dict]:
    """Normalize Workers AI response shapes into (text, usage)."""
    if not isinstance(data, dict):
        return str(data or ""), {}

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    result = data.get("result")
    if isinstance(result, dict):
        if isinstance(result.get("usage"), dict):
            usage = result["usage"] or usage
        if result.get("response"):
            return str(result["response"]), usage
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else {}
            if isinstance(msg, dict) and msg.get("content"):
                return str(msg["content"]), usage
        if result.get("output"):
            return str(result["output"]), usage

    if data.get("response"):
        return str(data["response"]), usage
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else {}
        if isinstance(msg, dict) and msg.get("content"):
            return str(msg["content"]), usage
    if data.get("output"):
        return str(data["output"]), usage

    # Some models return success wrapper without nested text.
    if data.get("success") is False:
        errors = data.get("errors") or []
        raise AdapterError(f"cloudflare error: {errors!r}"[:300])

    return "", usage
