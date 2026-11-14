from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generator, Optional

from app.adapters import ADAPTERS
from app.adapters.base import AdapterError, AdapterResult
from app.config import get_settings
from app.quality import looks_low_confidence
from app.registry import ModelSpec, estimate_cost, load_routing
from app.router import rank_models
from app.schemas import AttemptTrace


@dataclass
class CompletionOutcome:
    spec: ModelSpec
    result: AdapterResult
    est_cost: float
    fallback_used: bool
    fallback_from: str | None
    ranked_ids: list[str]
    attempts: list[AttemptTrace] = field(default_factory=list)
    error: str | None = None
    success: bool = True


def _call(
    spec: ModelSpec,
    prompt: str,
    *,
    messages: list[dict] | None = None,
) -> AdapterResult:
    adapter = ADAPTERS.get(spec.provider)
    if adapter is None:
        raise AdapterError(f"no adapter for provider {spec.provider}")
    timeout = load_routing().timeout_seconds
    return adapter.complete(
        api_model=spec.api_model,
        prompt=prompt,
        timeout=timeout,
        messages=messages,
    )


def iter_complete_with_fallback(
    prompt: str,
    task_type: str,
    *,
    force_model: str | None = None,
    ranked: list[ModelSpec] | None = None,
    allow_fallback: bool = True,
    messages: list[dict] | None = None,
) -> Generator[tuple[str, dict], None, CompletionOutcome]:
    """Yield attempt events, then return CompletionOutcome via StopIteration.value.

    When allow_fallback is False, only the first model is tried (strict baseline).
    """
    routing = load_routing()
    settings = get_settings()
    if ranked is None:
        ranked = [spec for spec, _ in rank_models(task_type)]
    if force_model:
        from app.registry import load_models

        catalog = load_models()
        if force_model in catalog:
            forced = catalog[force_model]
            ranked = [forced] + [m for m in ranked if m.id != force_model]
    ranked_ids = [m.id for m in ranked]
    max_attempts = 1 if not allow_fallback else routing.max_fallback_attempts
    attempt_specs = ranked[:max_attempts]
    last_error = "no models available"
    first_id = attempt_specs[0].id if attempt_specs else None
    traces: list[AttemptTrace] = []

    for i, spec in enumerate(attempt_specs):
        yield (
            "attempt_start",
            {"index": i, "model": spec.id, "provider": spec.provider},
        )
        try:
            result = _call(spec, prompt, messages=messages)
            cost = estimate_cost(spec, result.tokens_in, result.tokens_out)
            if looks_low_confidence(result.output):
                last_error = "low-confidence output"
                trace = AttemptTrace(
                    index=i,
                    model=spec.id,
                    provider=spec.provider,
                    outcome="low_confidence",
                    reason=last_error,
                    latency_ms=result.latency_ms,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    est_cost=round(cost, 6),
                )
                traces.append(trace)
                yield ("attempt_end", trace.model_dump())
                if not allow_fallback:
                    break
                continue
            trace = AttemptTrace(
                index=i,
                model=spec.id,
                provider=spec.provider,
                outcome="success",
                reason=None,
                latency_ms=result.latency_ms,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                est_cost=round(cost, 6),
            )
            traces.append(trace)
            yield ("attempt_end", trace.model_dump())
            return CompletionOutcome(
                spec=spec,
                result=result,
                est_cost=cost,
                fallback_used=i > 0,
                fallback_from=first_id if i > 0 else None,
                ranked_ids=ranked_ids,
                attempts=traces,
            )
        except Exception as exc:
            last_error = str(exc)
            trace = AttemptTrace(
                index=i,
                model=spec.id,
                provider=spec.provider,
                outcome="error",
                reason=last_error,
                latency_ms=0,
                tokens_in=0,
                tokens_out=0,
                est_cost=0.0,
            )
            traces.append(trace)
            yield ("attempt_end", trace.model_dump())
            if not allow_fallback:
                break
            continue

    if (
        allow_fallback
        and settings.allow_mock_fallback
        and (not attempt_specs or attempt_specs[0].provider != "mock")
    ):
        from app.registry import load_models

        mock = next(
            (
                model
                for model in load_models().values()
                if model.provider == "mock" and model.enabled
            ),
            None,
        )
        if mock is not None:
            yield (
                "attempt_start",
                {
                    "index": len(traces),
                    "model": mock.id,
                    "provider": mock.provider,
                },
            )
            result = _call(mock, prompt)
            cost = estimate_cost(mock, result.tokens_in, result.tokens_out)
            trace = AttemptTrace(
                index=len(traces),
                model=mock.id,
                provider=mock.provider,
                outcome="success",
                reason="mock_fallback",
                latency_ms=result.latency_ms,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                est_cost=round(cost, 6),
            )
            traces.append(trace)
            yield ("attempt_end", trace.model_dump())
            return CompletionOutcome(
                spec=mock,
                result=result,
                est_cost=cost,
                fallback_used=True,
                fallback_from=first_id,
                ranked_ids=ranked_ids,
                attempts=traces,
            )

    empty = AdapterResult(output="", tokens_in=0, tokens_out=0, latency_ms=0)
    from app.registry import load_models

    failed_spec = (
        attempt_specs[0] if attempt_specs else next(iter(load_models().values()))
    )
    return CompletionOutcome(
        spec=failed_spec,
        result=empty,
        est_cost=0.0,
        fallback_used=allow_fallback and len(traces) > 1,
        fallback_from=first_id if allow_fallback and len(traces) > 1 else None,
        ranked_ids=ranked_ids,
        attempts=traces,
        error=last_error,
        success=False,
    )


def complete_with_fallback(
    prompt: str,
    task_type: str,
    *,
    force_model: str | None = None,
    ranked: list[ModelSpec] | None = None,
    allow_fallback: bool = True,
    on_event: Optional[Callable[[str, dict], None]] = None,
    messages: list[dict] | None = None,
) -> CompletionOutcome:
    gen = iter_complete_with_fallback(
        prompt,
        task_type,
        force_model=force_model,
        ranked=ranked,
        allow_fallback=allow_fallback,
        messages=messages,
    )
    try:
        while True:
            event, payload = next(gen)
            if on_event is not None:
                on_event(event, payload)
    except StopIteration as stop:
        outcome = stop.value
        if outcome is None:
            raise RuntimeError("completion generator returned no outcome")
        return outcome
