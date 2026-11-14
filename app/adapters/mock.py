from __future__ import annotations

import hashlib
import time

from app.adapters.base import AdapterError, AdapterResult, ModelAdapter, normalize_messages
from app.classifier import classify


class MockAdapter(ModelAdapter):
    provider = "mock"

    def complete(
        self,
        *,
        api_model: str,
        prompt: str,
        timeout: float,
        messages: list | None = None,
    ) -> AdapterResult:
        chat = normalize_messages(prompt=prompt, messages=messages)
        last_user = next(
            (m["content"] for m in reversed(chat) if m["role"] == "user"),
            prompt,
        )
        blob = "\n".join(f"{m['role']}:{m['content']}" for m in chat)
        if "FAIL_ME" in last_user or "FAIL_ME" in blob:
            raise AdapterError("mock forced failure")
        start = time.perf_counter()
        task = classify(last_user)
        seed = int(hashlib.md5(blob.encode("utf-8")).hexdigest()[:8], 16)
        history_note = ""
        if len(chat) > 1:
            history_note = f"\n[context: {len(chat)} turns]"
        bodies = {
            "qa": f"[{api_model}] Direct answer: {last_user.split('?')[0].strip()[:80]}. Confidence is demo-grade.{history_note}",
            "code": f"[{api_model}]\n```python\n# demo implementation for: {last_user[:60]}\ndef solution():\n    return {seed % 97}\n```{history_note}",
            "summarization": f"[{api_model}] Summary: {last_user[:140]}… → cheaper model is enough for this compression task.{history_note}",
            "reasoning": f"[{api_model}] Step 1: restate the problem. Step 2: apply a simple invariant. Step 3: answer derived from prompt hash {seed % 10}.{history_note}",
            "creative": f"[{api_model}] In a dim server room, a router chose a cheaper muse.\nIt still wrote: {last_user[:50]}…{history_note}",
        }
        output = bodies.get(task, bodies["qa"])
        if api_model == "mock-strong":
            output += "\n[strong model extra check: answer reviewed]"
        tokens_in = max(8, len(blob) // 4)
        tokens_out = max(16, len(output) // 4)
        latency = {"mock-fast": 70, "mock-code": 160, "mock-strong": 380}.get(api_model, 120)
        elapsed = (time.perf_counter() - start) * 1000
        return AdapterResult(
            output=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=int(max(latency, elapsed)),
        )
