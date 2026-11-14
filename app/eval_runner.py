from __future__ import annotations

import json
import uuid
from typing import Generator

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import EvalRun
from app.engine import handle_complete
from app.schemas import CompleteRequest
from app.stats import eval_detail


def _status_line(policy: str, prompt_index: int, prompt_count: int) -> str:
    if policy == "baseline":
        return f"Calling baseline model for prompt {prompt_index} of {prompt_count}…"
    return f"Routing prompt {prompt_index} of {prompt_count}…"


def stream_eval(
    db: Session, limit: int | None = None, notes: str | None = None
) -> Generator[tuple[str, dict], None, None]:
    """Yield (event, data) for a fixture eval run. Ends with event=final."""
    path = get_settings().eval_fixture_path
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        prompts = prompts[:limit]

    run_id = str(uuid.uuid4())
    prompt_count = len(prompts)
    total_calls = prompt_count * 2
    db.add(EvalRun(id=run_id, prompt_count=prompt_count, notes=notes))
    db.commit()

    yield (
        "started",
        {
            "run_id": run_id,
            "prompt_count": prompt_count,
            "total_calls": total_calls,
            "status": f"Starting eval: {prompt_count} prompts × 2 policies ({total_calls} calls)…",
        },
    )

    call_index = 0
    for prompt_i, item in enumerate(prompts, start=1):
        hint = item.get("task_type")
        prompt = item["prompt"]
        preview = (prompt or "")[:120]
        for policy in ("baseline", "routed"):
            call_index += 1
            yield (
                "step",
                {
                    "index": call_index,
                    "total": total_calls,
                    "prompt_index": prompt_i,
                    "prompt_count": prompt_count,
                    "policy": policy,
                    "task_type": hint,
                    "prompt_preview": preview,
                    "status": _status_line(policy, prompt_i, prompt_count),
                },
            )
            result = handle_complete(
                db,
                CompleteRequest(prompt=prompt, task_type=hint, policy=policy),
                eval_run_id=run_id,
            )
            yield (
                "step_done",
                {
                    "index": call_index,
                    "total": total_calls,
                    "prompt_index": prompt_i,
                    "prompt_count": prompt_count,
                    "policy": policy,
                    "task_type": result.task_type,
                    "model": result.model,
                    "provider": result.provider,
                    "latency_ms": result.latency_ms,
                    "est_cost": result.est_cost,
                    "success": result.success,
                    "status": (
                        f"Finished {policy} · {result.model} · {result.latency_ms} ms"
                        if result.success
                        else f"Failed {policy} · {result.model}"
                    ),
                },
            )

    yield (
        "final",
        {"id": run_id, "prompt_count": prompt_count, **eval_detail(db, run_id)},
    )


def run_eval(
    db: Session, limit: int | None = None, notes: str | None = None
) -> dict:
    final: dict | None = None
    for event, payload in stream_eval(db, limit=limit, notes=notes):
        if event == "final":
            final = payload
    if final is None:
        raise RuntimeError("eval stream ended without final event")
    return final
