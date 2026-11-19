from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generator

from sqlalchemy.orm import Session

from app.adapters.base import AdapterResult
from app.chat_context import adapter_messages, decompose_input, resolve_prompt
from app.classifier import classify
from app.db import RequestLog
from app.decomposer import decompose, specialist_prompt
from app.synthesizer import clean_specialist_text
from app.fallback import CompletionOutcome, complete_with_fallback, iter_complete_with_fallback
from app.quality import quality_score
from app.registry import ModelSpec, load_models, load_routing
from app.router import (
    baseline_spec,
    diversify_ranked,
    rank_models,
    rank_models_detailed,
)
from app.schemas import (
    AttemptTrace,
    CompleteRequest,
    CompleteResponse,
    PartOutput,
    PromptPart,
    ScoreBreakdown,
)
from app.synthesizer import stitch

QUALITY_LABEL = "simple quality score"
ORCHESTRA_CONCURRENCY = 8


def _normalize_body(body: CompleteRequest) -> CompleteRequest:
    """Ensure body.prompt is the latest user turn for logging and legacy call sites."""
    prompt = resolve_prompt(body)
    if body.prompt == prompt:
        return body
    return body.model_copy(update={"prompt": prompt})


def _scoreboard_for(task_type: str) -> list[ScoreBreakdown]:
    return rank_models_detailed(task_type, include_unavailable=True)


def _resolve_ranked_for_task(
    body: CompleteRequest,
    task_type: str,
    *,
    used_providers: set[str] | None = None,
) -> tuple[list[ModelSpec], bool, str | None]:
    catalog = load_models()
    force = body.force_model if body.force_model in catalog else None

    if body.policy == "baseline" and not force:
        return [baseline_spec()], False, None

    scored = rank_models(task_type)
    if force:
        forced = catalog[force]
        scored = [(forced, 1.0)] + [(m, s) for m, s in scored if m.id != force]
        ranked = [m for m, _ in scored]
        return ranked, True, force

    if body.policy == "orchestra" and used_providers is not None:
        ranked = diversify_ranked(scored, used_providers)
    else:
        ranked = [spec for spec, _ in scored]
    return ranked, True, force


def _assign_orchestra_rankings(
    body: CompleteRequest, parts: list[PromptPart]
) -> dict[str, list[ModelSpec]]:
    """Pick per-part ranked lists with provider diversity across parts."""
    used: set[str] = set()
    assigned: dict[str, list[ModelSpec]] = {}
    for part in parts:
        ranked, _, _ = _resolve_ranked_for_task(
            body, part.task_type, used_providers=used
        )
        assigned[part.id] = ranked
        if ranked:
            used.add(ranked[0].provider)
    return assigned


def _build_response(
    body: CompleteRequest,
    task_type: str,
    outcome: CompletionOutcome,
    scoreboard: list[ScoreBreakdown],
    q: float | None,
    *,
    parts: list[PromptPart] | None = None,
    part_outputs: list[PartOutput] | None = None,
    synthesis_method: str = "none",
) -> CompleteResponse:
    return CompleteResponse(
        output=outcome.result.output,
        task_type=task_type,
        model=outcome.spec.id,
        provider=outcome.spec.provider,
        policy=body.policy,
        tokens_in=outcome.result.tokens_in,
        tokens_out=outcome.result.tokens_out,
        est_cost=round(outcome.est_cost, 6),
        latency_ms=outcome.result.latency_ms,
        success=outcome.success,
        fallback_used=outcome.fallback_used,
        fallback_from=outcome.fallback_from,
        quality_score=None if q is None else round(q, 4),
        quality_label=QUALITY_LABEL,
        ranked=outcome.ranked_ids,
        attempts=outcome.attempts,
        scoreboard=scoreboard,
        parts=parts or [],
        part_outputs=part_outputs or [],
        synthesis_method=synthesis_method,  # type: ignore[arg-type]
    )


def _persist(
    db: Session,
    body: CompleteRequest,
    task_type: str,
    outcome: CompletionOutcome,
    q: float | None,
    *,
    eval_run_id: str | None,
) -> None:
    preview = outcome.result.output[:500]
    row = RequestLog(
        prompt=body.prompt,
        task_type=task_type,
        policy=body.policy,
        model=outcome.spec.id,
        provider=outcome.spec.provider,
        tokens_in=outcome.result.tokens_in,
        tokens_out=outcome.result.tokens_out,
        est_cost=outcome.est_cost,
        latency_ms=outcome.result.latency_ms,
        success=outcome.success,
        fallback_used=outcome.fallback_used,
        fallback_from=outcome.fallback_from,
        quality_score=0.0 if q is None else q,
        error=outcome.error,
        response_preview=preview,
        eval_run_id=eval_run_id,
    )
    db.add(row)
    db.commit()


def _quality_for(outcome: CompletionOutcome, prompt: str, task_type: str) -> float | None:
    if not outcome.success:
        return None
    return quality_score(
        prompt,
        outcome.result.output,
        task_type,
        outcome.spec.strengths.get(task_type, 0.5),
    )


def _tag_attempts(attempts: list[AttemptTrace], part_id: str) -> list[AttemptTrace]:
    tagged: list[AttemptTrace] = []
    for a in attempts:
        tagged.append(a.model_copy(update={"part_id": part_id}))
    return tagged


def _run_part(
    body: CompleteRequest,
    part: PromptPart,
    *,
    original: str,
    siblings: list[str],
    ranked: list[ModelSpec] | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> PartOutput:
    if ranked is None:
        ranked, allow_fallback, force = _resolve_ranked_for_task(
            body, part.task_type, used_providers=set()
        )
    else:
        allow_fallback = True
        force = body.force_model if body.force_model in load_models() else None
    specialist = specialist_prompt(
        part, original=original, sibling_summaries=siblings
    )
    scoreboard = _scoreboard_for(part.task_type)
    msgs = adapter_messages(body, override_user=specialist)
    outcome = complete_with_fallback(
        specialist,
        part.task_type,
        force_model=force,
        ranked=ranked,
        allow_fallback=allow_fallback,
        messages=msgs,
        on_event=on_event,
    )
    raw_out = outcome.result.output if outcome.success else ""
    return PartOutput(
        part_id=part.id,
        task_type=part.task_type,
        text=clean_specialist_text(raw_out) if raw_out else "",
        model=outcome.spec.id,
        provider=outcome.spec.provider,
        success=outcome.success,
        tokens_in=outcome.result.tokens_in,
        tokens_out=outcome.result.tokens_out,
        est_cost=round(outcome.est_cost, 6),
        latency_ms=outcome.result.latency_ms,
        attempts=_tag_attempts(outcome.attempts, part.id),
        ranked=outcome.ranked_ids,
        scoreboard=scoreboard,
        error=outcome.error,
    )


def _aggregate_orchestra(
    body: CompleteRequest,
    parts: list[PromptPart],
    part_outputs: list[PartOutput],
    wall_ms: int,
) -> tuple[CompletionOutcome, list[ScoreBreakdown], float | None, str]:
    combined, method = stitch(parts, part_outputs)
    tokens_in = sum(o.tokens_in for o in part_outputs)
    tokens_out = sum(o.tokens_out for o in part_outputs)
    est_cost = sum(o.est_cost for o in part_outputs)
    any_success = any(o.success for o in part_outputs)
    all_success = all(o.success for o in part_outputs) and bool(part_outputs)
    models_used = [o.model for o in part_outputs if o.success]
    primary = next((o for o in part_outputs if o.success), None)
    catalog = load_models()
    if primary and primary.model in catalog:
        spec = catalog[primary.model]
    else:
        # Synthetic placeholder when everything failed — use baseline for logging shape.
        spec = baseline_spec()

    attempts: list[AttemptTrace] = []
    for o in part_outputs:
        attempts.extend(o.attempts)

    fallback_used = any(
        len(o.attempts) > 1 for o in part_outputs
    ) or any(not o.success for o in part_outputs if primary and o.part_id != primary.part_id)

    outcome = CompletionOutcome(
        result=AdapterResult(
            output=combined if any_success else "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=wall_ms,
        ),
        spec=spec,
        est_cost=est_cost,
        success=any_success,
        fallback_used=fallback_used,
        fallback_from=None,
        ranked_ids=models_used,
        attempts=attempts,
        error=None if any_success else "all specialist parts failed",
    )

    # Union scoreboard: primary part's board, or first part.
    scoreboard = part_outputs[0].scoreboard if part_outputs else []
    task_label = "+".join(sorted({p.task_type for p in parts})) or "qa"
    q = None
    if any_success:
        # Average proxy across successful parts, weighted by strength priors.
        scores = []
        for o in part_outputs:
            if not o.success:
                continue
            strength = 0.5
            if o.model in catalog:
                strength = catalog[o.model].strengths.get(o.task_type, 0.5)
            scores.append(quality_score(body.prompt, o.text, o.task_type, strength))
        q = sum(scores) / len(scores) if scores else None
        if all_success and len(parts) > 1:
            q = min(1.0, (q or 0) + 0.03)

    return outcome, scoreboard, q, task_label if len(parts) > 1 else parts[0].task_type


def handle_orchestra(
    db: Session, body: CompleteRequest, *, eval_run_id: str | None = None
) -> CompleteResponse:
    body = _normalize_body(body)
    parts = decompose(decompose_input(body), body.task_type)
    sibling_summaries = [
        f"{p.id} [{p.task_type}]: {p.text[:80]}{'…' if len(p.text) > 80 else ''}"
        for p in parts
    ]
    started = time.perf_counter()
    part_outputs: list[PartOutput] = []
    rankings = _assign_orchestra_rankings(body, parts)

    def job(part: PromptPart) -> PartOutput:
        siblings = [s for s in sibling_summaries if not s.startswith(f"{part.id} ")]
        return _run_part(
            body,
            part,
            original=body.prompt,
            siblings=siblings,
            ranked=rankings.get(part.id),
        )

    with ThreadPoolExecutor(max_workers=min(ORCHESTRA_CONCURRENCY, max(1, len(parts)))) as pool:
        futs = {pool.submit(job, p): p.id for p in parts}
        by_id: dict[str, PartOutput] = {}
        for fut in as_completed(futs):
            out = fut.result()
            by_id[out.part_id] = out
        part_outputs = [by_id[p.id] for p in parts if p.id in by_id]

    wall_ms = int((time.perf_counter() - started) * 1000)
    outcome, scoreboard, q, task_label = _aggregate_orchestra(
        body, parts, part_outputs, wall_ms
    )
    _persist(db, body, task_label, outcome, q, eval_run_id=eval_run_id)
    return _build_response(
        body,
        task_label,
        outcome,
        scoreboard,
        q,
        parts=parts,
        part_outputs=part_outputs,
        synthesis_method="stitch",
    )


def stream_orchestra(
    db: Session, body: CompleteRequest
) -> Generator[tuple[str, dict], None, None]:
    """Stream orchestra events while running specialty parts in parallel."""
    body = _normalize_body(body)
    parts = decompose(decompose_input(body), body.task_type)
    yield (
        "decomposed",
        {
            "parts": [p.model_dump() for p in parts],
            "count": len(parts),
        },
    )

    sibling_summaries = [
        f"{p.id} [{p.task_type}]: {p.text[:80]}{'…' if len(p.text) > 80 else ''}"
        for p in parts
    ]
    rankings = _assign_orchestra_rankings(body, parts)

    # Emit all rankings first so the UI can show preferred models immediately.
    for part in parts:
        ranked = rankings.get(part.id) or []
        scored = rank_models(part.task_type)
        raw_top_id = scored[0][0].id if scored else None
        scoreboard = _scoreboard_for(part.task_type)
        yield (
            "part_ranked",
            {
                "part_id": part.id,
                "task_type": part.task_type,
                "candidates": [row.model_dump() for row in scoreboard],
                "preferred_model": ranked[0].id if ranked else None,
                "preferred_provider": ranked[0].provider if ranked else None,
                "diversity_applied": bool(
                    ranked and raw_top_id and ranked[0].id != raw_top_id
                ),
            },
        )

    started = time.perf_counter()
    event_q: queue.Queue[tuple[str, dict] | None] = queue.Queue()
    workers = min(ORCHESTRA_CONCURRENCY, max(1, len(parts)))

    def worker(part: PromptPart) -> None:
        siblings = [s for s in sibling_summaries if not s.startswith(f"{part.id} ")]

        def on_event(event: str, payload: dict) -> None:
            enriched = dict(payload)
            enriched["part_id"] = part.id
            if event == "attempt_start":
                event_q.put(("part_attempt_start", enriched))
            elif event == "attempt_end":
                event_q.put(("part_attempt_end", enriched))

        try:
            part_out = _run_part(
                body,
                part,
                original=body.prompt,
                siblings=siblings,
                ranked=rankings.get(part.id),
                on_event=on_event,
            )
            event_q.put(("part_done", part_out.model_dump()))
        except Exception as exc:  # noqa: BLE001 — surface to stream consumer
            event_q.put(
                (
                    "part_done",
                    PartOutput(
                        part_id=part.id,
                        task_type=part.task_type,
                        text="",
                        model="",
                        provider="",
                        success=False,
                        error=str(exc),
                    ).model_dump(),
                )
            )

    part_outputs: list[PartOutput] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, p) for p in parts]
        done_parts = 0
        while done_parts < len(parts):
            try:
                item = event_q.get(timeout=0.05)
            except queue.Empty:
                # Propagate worker crashes early if the queue is idle.
                for fut in futures:
                    if fut.done() and fut.exception() is not None:
                        fut.result()
                continue
            if item is None:
                continue
            event_name, payload = item
            yield (event_name, payload)
            if event_name == "part_done":
                done_parts += 1
                part_outputs.append(PartOutput.model_validate(payload))
        for fut in futures:
            fut.result()

    by_id = {o.part_id: o for o in part_outputs}
    part_outputs = [by_id[p.id] for p in parts if p.id in by_id]

    wall_ms = int((time.perf_counter() - started) * 1000)
    outcome, scoreboard, q, task_label = _aggregate_orchestra(
        body, parts, part_outputs, wall_ms
    )
    combined = outcome.result.output
    yield (
        "synthesized",
        {
            "synthesis_method": "stitch",
            "output_preview": combined[:280],
            "part_count": len(parts),
        },
    )
    yield (
        "quality",
        {
            "quality_score": None if q is None else round(q, 4),
            "quality_label": QUALITY_LABEL,
            "success": outcome.success,
        },
    )
    _persist(db, body, task_label, outcome, q, eval_run_id=None)
    response = _build_response(
        body,
        task_label,
        outcome,
        scoreboard,
        q,
        parts=parts,
        part_outputs=part_outputs,
        synthesis_method="stitch",
    )
    yield ("final", response.model_dump())


def handle_complete(
    db: Session, body: CompleteRequest, *, eval_run_id: str | None = None
) -> CompleteResponse:
    body = _normalize_body(body)
    if body.policy == "orchestra":
        return handle_orchestra(db, body, eval_run_id=eval_run_id)

    task_type = classify(decompose_input(body), body.task_type)
    scoreboard = _scoreboard_for(task_type)
    ranked, allow_fallback, force = _resolve_ranked_for_task(body, task_type)
    msgs = adapter_messages(body)
    outcome = complete_with_fallback(
        body.prompt,
        task_type,
        force_model=force,
        ranked=ranked,
        allow_fallback=allow_fallback,
        messages=msgs,
    )
    q = _quality_for(outcome, body.prompt, task_type)
    _persist(db, body, task_type, outcome, q, eval_run_id=eval_run_id)
    return _build_response(body, task_type, outcome, scoreboard, q)


def stream_complete(
    db: Session, body: CompleteRequest
) -> Generator[tuple[str, dict], None, None]:
    """Yield (event_name, payload) including live attempt_start before each provider call."""
    body = _normalize_body(body)
    if body.policy == "orchestra":
        yield from stream_orchestra(db, body)
        return

    task_type = classify(decompose_input(body), body.task_type)
    yield ("classified", {"task_type": task_type, "hint": body.task_type})

    scoreboard = _scoreboard_for(task_type)
    yield (
        "ranked",
        {
            "task_type": task_type,
            "candidates": [row.model_dump() for row in scoreboard],
        },
    )

    ranked, allow_fallback, force = _resolve_ranked_for_task(body, task_type)
    msgs = adapter_messages(body)
    gen = iter_complete_with_fallback(
        body.prompt,
        task_type,
        force_model=force,
        ranked=ranked,
        allow_fallback=allow_fallback,
        messages=msgs,
    )
    outcome: CompletionOutcome | None = None
    try:
        while True:
            event, payload = next(gen)
            yield (event, payload)
    except StopIteration as stop:
        outcome = stop.value

    if outcome is None:
        raise RuntimeError("stream completion produced no outcome")

    q = _quality_for(outcome, body.prompt, task_type)
    yield (
        "quality",
        {
            "quality_score": None if q is None else round(q, 4),
            "quality_label": QUALITY_LABEL,
            "success": outcome.success,
        },
    )
    _persist(db, body, task_type, outcome, q, eval_run_id=None)
    response = _build_response(body, task_type, outcome, scoreboard, q)
    yield ("final", response.model_dump())
