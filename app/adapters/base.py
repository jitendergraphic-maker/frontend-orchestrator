from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AdapterResult:
    output: str
    tokens_in: int
    tokens_out: int
    latency_ms: int


class AdapterError(Exception):
    pass


def normalize_messages(
    *, prompt: str, messages: list[dict[str, Any]] | None
) -> list[dict[str, str]]:
    if messages:
        out: list[dict[str, str]] = []
        for m in messages:
            role = str(m.get("role") or "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            content = str(m.get("content") or "")
            if content:
                out.append({"role": role, "content": content})
        if out:
            return out
    return [{"role": "user", "content": prompt}]


class ModelAdapter:
    provider: str

    def complete(
        self,
        *,
        api_model: str,
        prompt: str,
        timeout: float,
        messages: list[dict[str, Any]] | None = None,
    ) -> AdapterResult:
        raise NotImplementedError
