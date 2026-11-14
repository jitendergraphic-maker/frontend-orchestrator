from __future__ import annotations

from app.config import get_settings
from app.registry import (
    ModelSpec,
    RoutingConfig,
    load_models,
    load_routing,
    routing_cost_per_1k,
)
from app.schemas import ScoreBreakdown


def provider_status(spec: ModelSpec) -> tuple[bool, str | None]:
    """Return (available, reject_reason). Never includes secret values."""
    settings = get_settings()
    if not spec.enabled:
        return False, "disabled"
    if spec.provider == "mock":
        return False, "mock_excluded"
    if spec.provider == "groq":
        if not settings.groq_api_key:
            return False, "missing_credentials"
        return True, None
    if spec.provider == "nvidia":
        if not settings.nvidia_api_key:
            return False, "missing_credentials"
        return True, None
    if spec.provider == "openrouter":
        if not settings.openrouter_api_key:
            return False, "missing_credentials"
        return True, None
    if spec.provider == "gemini":
        if not settings.gemini_api_key:
            return False, "missing_credentials"
        return True, None
    if spec.provider == "coarena":
        if not settings.coarena_api_key:
            return False, "missing_credentials"
        return True, None
    if spec.provider == "ollama":
        return False, "ollama_disabled"
    if spec.provider in ("openai", "anthropic"):
        return False, "unsupported_for_routing"
    return False, "unsupported_provider"


def provider_available(spec: ModelSpec) -> bool:
    available, _ = provider_status(spec)
    return available


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]. Zero-range → 0.5 for every entry (no NaN)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _tie_key(spec: ModelSpec, score: float) -> tuple:
    """Deterministic tie-break: higher score, then lower cost, then id."""
    return (-score, routing_cost_per_1k(spec), spec.id)


def rank_models_detailed(
    task_type: str, *, include_unavailable: bool = False
) -> list[ScoreBreakdown]:
    """Score eligible models and optionally include rejected ones with reasons.

    Formula (documented):
      score = w_q * strength[task] - w_c * cost_norm - w_l * latency_norm
    where cost_norm / latency_norm are min-max over the *eligible* candidate set
    (or over all non-mock enabled models when include_unavailable is True for
    scoring the eligible subset only — rejected models get score 0).
    """
    routing = load_routing()
    catalog = [
        model
        for model in load_models().values()
        if model.enabled and model.provider != "mock"
    ]
    statuses = {m.id: provider_status(m) for m in catalog}
    eligible = [m for m in catalog if statuses[m.id][0]]
    pool = catalog if include_unavailable else eligible
    if not eligible:
        # Still return rejected entries for explainability when requested
        if include_unavailable:
            return [
                ScoreBreakdown(
                    id=m.id,
                    provider=m.provider,
                    display_name=m.display_name,
                    strength=float(m.strengths.get(task_type, 0.5)),
                    cost_norm=0.0,
                    latency_norm=0.0,
                    score=0.0,
                    eligible=False,
                    reject_reason=statuses[m.id][1],
                )
                for m in catalog
            ]
        return []

    costs = [routing_cost_per_1k(m) for m in eligible]
    lats = [float(m.typical_latency_ms) for m in eligible]
    cost_n = _normalize(costs)
    lat_n = _normalize(lats)
    norms = {
        m.id: (c, l) for m, c, l in zip(eligible, cost_n, lat_n)
    }

    scored: list[ScoreBreakdown] = []
    for spec in pool:
        strength = float(spec.strengths.get(task_type, 0.5))
        available, reason = statuses[spec.id]
        if not available:
            scored.append(
                ScoreBreakdown(
                    id=spec.id,
                    provider=spec.provider,
                    display_name=spec.display_name,
                    strength=strength,
                    cost_norm=0.0,
                    latency_norm=0.0,
                    score=0.0,
                    eligible=False,
                    reject_reason=reason,
                )
            )
            continue
        c, l = norms[spec.id]
        score = routing.quality * strength - routing.cost * c - routing.latency * l
        if score != score:  # NaN guard
            score = 0.0
        scored.append(
            ScoreBreakdown(
                id=spec.id,
                provider=spec.provider,
                display_name=spec.display_name,
                strength=round(strength, 4),
                cost_norm=round(c, 4),
                latency_norm=round(l, 4),
                score=round(score, 4),
                eligible=True,
                reject_reason=None,
            )
        )

    scored.sort(
        key=lambda row: (
            0 if row.eligible else 1,
            *_tie_key(
                next(m for m in catalog if m.id == row.id),
                row.score,
            ),
        )
    )
    return scored


def rank_models(
    task_type: str, *, include_unavailable: bool = False
) -> list[tuple[ModelSpec, float]]:
    models = load_models()
    detailed = rank_models_detailed(task_type, include_unavailable=include_unavailable)
    out: list[tuple[ModelSpec, float]] = []
    for row in detailed:
        if not row.eligible and not include_unavailable:
            continue
        if not row.eligible:
            continue
        out.append((models[row.id], row.score))
    return out


def diversify_ranked(
    ranked: list[tuple[ModelSpec, float]],
    used_providers: set[str],
    *,
    margin: float | None = None,
) -> list[ModelSpec]:
    """Reorder so the first pick prefers an unused provider when close in score.

    If the top model’s provider is already used by another orchestra part, and
    another provider’s score is within ``margin`` of the top score, promote that
    model to the front. Fallback order otherwise stays score-sorted.
    """
    if not ranked:
        return []
    if margin is None:
        margin = load_routing().diversity_score_margin
    top_spec, top_score = ranked[0]
    if not used_providers or top_spec.provider not in used_providers:
        return [spec for spec, _ in ranked]

    alt: ModelSpec | None = None
    for spec, score in ranked[1:]:
        if spec.provider in used_providers:
            continue
        if (top_score - score) <= margin + 1e-12:
            alt = spec
            break
    if alt is None:
        return [spec for spec, _ in ranked]
    rest = [spec for spec, _ in ranked if spec.id != alt.id]
    return [alt] + rest


def baseline_spec() -> ModelSpec:
    routing: RoutingConfig = load_routing()
    models = load_models()
    spec = models.get(routing.baseline_model)
    if spec is None:
        raise RuntimeError("baseline model missing from registry")
    return spec
