from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator

from sqlalchemy.orm import Session

from app.adapters.base import AdapterResult
from app.classifier import classify
from app.db import RequestLog
from app.decomposer import decompose, specialist_prompt
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

QUALITY_LABEL = "heuristic quality proxy"
ORCHESTRA_CONCURRENCY = 4


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
) -> PartOutput:
    if ranked is None:
        ranked, allow_fallback, force = _resolve_ranked_for_task(
            body, part.task_type, used_providers=set()
        )
    else:
        allow_fallback = True
        force = body.force_model if body.force_model in load_models() else None
    prompt = specialist_prompt(
        part, original=original, sibling_summaries=siblings
    )
    scoreboard = _scoreboard_for(part.task_type)
    outcome = complete_with_fallback(
        prompt,
        part.task_type,
        force_model=force,
        ranked=ranked,
        allow_fallback=allow_fallback,
    )
    return PartOutput(
        part_id=part.id,
        task_type=part.task_type,
        text=outcome.result.output if outcome.success else "",
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
    parts = decompose(body.prompt, body.task_type)
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
    parts = decompose(body.prompt, body.task_type)
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
    started = time.perf_counter()
    part_outputs: list[PartOutput] = []
    used_providers: set[str] = set()

    # Sequential in stream path so theater can show live part events;
    # still honest timing (no fake parallel animation).
    for part in parts:
        siblings = [s for s in sibling_summaries if not s.startswith(f"{part.id} ")]
        scored = rank_models(part.task_type)
        raw_top_id = scored[0][0].id if scored else None
        ranked, allow_fallback, force = _resolve_ranked_for_task(
            body, part.task_type, used_providers=used_providers
        )
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
        prompt = specialist_prompt(
            part, original=body.prompt, sibling_summaries=siblings
        )
        if ranked:
            used_providers.add(ranked[0].provider)
        gen = iter_complete_with_fallback(
            prompt,
            part.task_type,
            force_model=force,
            ranked=ranked,
            allow_fallback=allow_fallback,
        )
        outcome: CompletionOutcome | None = None
        try:
            while True:
                event, payload = next(gen)
                enriched = dict(payload)
                enriched["part_id"] = part.id
                if event == "attempt_start":
                    yield ("part_attempt_start", enriched)
                elif event == "attempt_end":
                    yield ("part_attempt_end", enriched)
                else:
                    yield (event, enriched)
        except StopIteration as stop:
            outcome = stop.value
        if outcome is None:
            raise RuntimeError(f"part {part.id} produced no outcome")

        part_out = PartOutput(
            part_id=part.id,
            task_type=part.task_type,
            text=outcome.result.output if outcome.success else "",
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
        part_outputs.append(part_out)
        yield ("part_done", part_out.model_dump())

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
    if body.policy == "orchestra":
        return handle_orchestra(db, body, eval_run_id=eval_run_id)

    task_type = classify(body.prompt, body.task_type)
    scoreboard = _scoreboard_for(task_type)
    ranked, allow_fallback, force = _resolve_ranked_for_task(body, task_type)
    outcome = complete_with_fallback(
        body.prompt,
        task_type,
        force_model=force,
        ranked=ranked,
        allow_fallback=allow_fallback,
    )
    q = _quality_for(outcome, body.prompt, task_type)
    _persist(db, body, task_type, outcome, q, eval_run_id=eval_run_id)
    return _build_response(body, task_type, outcome, scoreboard, q)


def stream_complete(
    db: Session, body: CompleteRequest
) -> Generator[tuple[str, dict], None, None]:
    """Yield (event_name, payload) including live attempt_start before each provider call."""
    if body.policy == "orchestra":
        yield from stream_orchestra(db, body)
        return

    task_type = classify(body.prompt, body.task_type)
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
    gen = iter_complete_with_fallback(
        body.prompt,
        task_type,
        force_model=force,
        ranked=ranked,
        allow_fallback=allow_fallback,
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
