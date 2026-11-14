from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import case, func
from sqlalchemy.orm import Query, Session

from app.db import EvalRun, RequestLog
from app.registry import estimate_cost, load_models
from app.router import baseline_spec

ROUTED_POLICIES = ("orchestra", "routed")
BASELINE_POLICIES = ("baseline",)


def _apply_live_ops(q: Query) -> Query:
    """Operational Overview: exclude mock providers and fixture eval rows."""
    return q.filter(RequestLog.provider != "mock").filter(RequestLog.eval_run_id.is_(None))


def _live_ops_traffic_q(db: Session) -> Query:
    """Live orchestra + routed rows (the traffic we compare against locked baseline)."""
    q = db.query(RequestLog).filter(RequestLog.policy.in_(list(ROUTED_POLICIES)))
    return _apply_live_ops(q)


def _counterfactual_baseline_from_rows(rows: list[RequestLog]) -> dict:
    """What the locked expensive baseline would have cost for the same tokens."""
    spec = baseline_spec()
    n = len(rows)
    if n == 0:
        return {
            "requests": 0,
            "total_cost": 0.0,
            "avg_latency_ms": float(spec.typical_latency_ms),
            "p50_latency_ms": float(spec.typical_latency_ms),
            "avg_quality": 0.0,
            "success_rate": 0.0,
            "fallback_rate": 0.0,
            "cost_basis": "counterfactual",
            "baseline_model": spec.id,
        }
    total_cost = 0.0
    for r in rows:
        total_cost += estimate_cost(spec, int(r.tokens_in or 0), int(r.tokens_out or 0))
    avg_lat = sum(int(r.latency_ms or 0) for r in rows) / n
    # Quality/success still come from actual traffic outcomes (same prompts).
    avg_q = sum(float(r.quality_score or 0) for r in rows) / n
    success = sum(1.0 if r.success else 0.0 for r in rows) / n
    fallback = sum(1.0 if r.fallback_used else 0.0 for r in rows) / n
    return {
        "requests": n,
        "total_cost": round(total_cost, 6),
        "avg_latency_ms": round(float(spec.typical_latency_ms), 1),
        "p50_latency_ms": round(float(spec.typical_latency_ms), 1),
        "avg_quality": round(avg_q, 4),
        "success_rate": round(success, 4),
        "fallback_rate": round(fallback, 4),
        "cost_basis": "counterfactual",
        "baseline_model": spec.id,
    }


def _safe_agg(
    db: Session,
    *,
    policies: tuple[str, ...] | list[str] | None = None,
    policy: str | None = None,
    eval_run_id: str | None = None,
    live_ops: bool = False,
) -> dict:
    q = db.query(
        func.count(RequestLog.id).label("n"),
        func.coalesce(func.sum(RequestLog.est_cost), 0.0).label("cost"),
        func.coalesce(func.avg(RequestLog.latency_ms), 0.0).label("avg_latency"),
        func.coalesce(func.avg(RequestLog.quality_score), 0.0).label("quality"),
        func.avg(case((RequestLog.success.is_(True), 1.0), else_=0.0)).label("success_rate"),
        func.avg(case((RequestLog.fallback_used.is_(True), 1.0), else_=0.0)).label(
            "fallback_rate"
        ),
    )
    if policy:
        q = q.filter(RequestLog.policy == policy)
    elif policies:
        q = q.filter(RequestLog.policy.in_(list(policies)))
    if eval_run_id:
        q = q.filter(RequestLog.eval_run_id == eval_run_id)
    if live_ops:
        q = _apply_live_ops(q)
    row = q.one()
    n = int(row.n or 0)
    avg_lat = float(row.avg_latency or 0)
    return {
        "requests": n,
        "total_cost": round(float(row.cost or 0), 6),
        "avg_latency_ms": round(avg_lat, 1),
        "p50_latency_ms": round(avg_lat, 1),
        "avg_quality": round(float(row.quality or 0), 4),
        "success_rate": round(float(row.success_rate or 0), 4),
        "fallback_rate": round(float(row.fallback_rate or 0), 4),
    }


def overview(
    db: Session,
    eval_run_id: str | None = None,
    *,
    scope: str = "live",
) -> dict:
    as_of = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # Matched fixture eval: routed (+ orchestra if any) vs baseline for that run.
    if eval_run_id:
        routed = _safe_agg(
            db, policies=("routed", "orchestra"), eval_run_id=eval_run_id
        )
        baseline = _safe_agg(db, policies=BASELINE_POLICIES, eval_run_id=eval_run_id)
        savings = None
        if baseline["total_cost"] > 0:
            savings = round(
                (baseline["total_cost"] - routed["total_cost"]) / baseline["total_cost"],
                4,
            )
        return {
            "routed": routed,
            "baseline": baseline,
            "cost_savings_vs_baseline": savings,
            "scope": "eval",
            "as_of": as_of,
        }

    live_ops = scope != "all"
    if live_ops:
        # Routed/orchestra = actual live spend.
        # Baseline cost = locked model list price on the SAME live tokens
        # (old cheap baseline policy rows must not undercut the strip).
        routed = _safe_agg(db, policies=ROUTED_POLICIES, live_ops=True)
        traffic = _live_ops_traffic_q(db).all()
        baseline = _counterfactual_baseline_from_rows(traffic)
        scope_label = "live"
    else:
        # Legacy: all rows for routed / baseline policies (includes mock + eval).
        routed = _safe_agg(db, policy="routed")
        baseline = _safe_agg(db, policy="baseline")
        scope_label = "all"

    savings = None
    if baseline["total_cost"] > 0:
        savings = round(
            (baseline["total_cost"] - routed["total_cost"]) / baseline["total_cost"],
            4,
        )
    return {
        "routed": routed,
        "baseline": baseline,
        "cost_savings_vs_baseline": savings,
        "scope": scope_label,
        "as_of": as_of,
        "baseline_model": baseline.get("baseline_model") or baseline_spec().id,
    }


def by_model(
    db: Session,
    *,
    include_retired: bool = False,
    scope: str = "live",
) -> list[dict]:
    """Traffic rollup merged with the live catalog.

    Catalog models always appear (0 requests if unused). Retired ids from old
    logs are omitted unless include_retired is True.
    """
    q = db.query(
        RequestLog.model,
        RequestLog.provider,
        func.count(RequestLog.id),
        func.coalesce(func.sum(RequestLog.est_cost), 0.0),
        func.coalesce(func.avg(RequestLog.latency_ms), 0.0),
        func.coalesce(func.avg(RequestLog.quality_score), 0.0),
        func.avg(case((RequestLog.success.is_(True), 1.0), else_=0.0)),
    )
    if scope != "all":
        q = _apply_live_ops(q)
    rows = q.group_by(RequestLog.model, RequestLog.provider).all()

    traffic: dict[str, dict] = {}
    for r in rows:
        mid = str(r[0] or "")
        traffic[mid] = {
            "model": mid,
            "provider": r[1],
            "requests": int(r[2]),
            "total_cost": round(float(r[3]), 6),
            "avg_latency_ms": round(float(r[4]), 1),
            "avg_quality": round(float(r[5]), 4),
            "success_rate": round(float(r[6] or 0), 4),
        }

    catalog = {
        mid: spec
        for mid, spec in load_models().items()
        if spec.enabled and spec.provider != "mock"
    }
    out: list[dict] = []
    for mid, spec in catalog.items():
        hit = traffic.get(mid)
        if hit:
            out.append(
                {
                    **hit,
                    "provider": spec.provider,
                    "display_name": spec.display_name,
                    "in_catalog": True,
                }
            )
        else:
            out.append(
                {
                    "model": mid,
                    "provider": spec.provider,
                    "display_name": spec.display_name,
                    "requests": 0,
                    "total_cost": 0.0,
                    "avg_latency_ms": 0.0,
                    "avg_quality": 0.0,
                    "success_rate": 0.0,
                    "in_catalog": True,
                }
            )

    if include_retired:
        for mid, hit in traffic.items():
            if mid in catalog:
                continue
            out.append({**hit, "display_name": mid, "in_catalog": False})

    out.sort(key=lambda row: (-int(row["requests"]), str(row["model"])))
    return out


def by_task(
    db: Session,
    *,
    scope: str = "live",
    eval_run_id: str | None = None,
) -> list[dict]:
    # Matched eval: real baseline + routed/orchestra rows for that run.
    if eval_run_id:
        q = db.query(
            RequestLog.task_type,
            RequestLog.policy,
            func.count(RequestLog.id),
            func.coalesce(func.sum(RequestLog.est_cost), 0.0),
            func.coalesce(func.avg(RequestLog.latency_ms), 0.0),
            func.coalesce(func.avg(RequestLog.quality_score), 0.0),
        ).filter(RequestLog.eval_run_id == eval_run_id)
        rows = q.group_by(RequestLog.task_type, RequestLog.policy).all()
        return [
            {
                "task_type": r[0],
                "policy": r[1],
                "requests": int(r[2]),
                "total_cost": round(float(r[3]), 6),
                "avg_latency_ms": round(float(r[4]), 1),
                "avg_quality": round(float(r[5]), 4),
            }
            for r in rows
        ]

    if scope == "all":
        q = db.query(
            RequestLog.task_type,
            RequestLog.policy,
            func.count(RequestLog.id),
            func.coalesce(func.sum(RequestLog.est_cost), 0.0),
            func.coalesce(func.avg(RequestLog.latency_ms), 0.0),
            func.coalesce(func.avg(RequestLog.quality_score), 0.0),
        )
        rows = q.group_by(RequestLog.task_type, RequestLog.policy).all()
        return [
            {
                "task_type": r[0],
                "policy": r[1],
                "requests": int(r[2]),
                "total_cost": round(float(r[3]), 6),
                "avg_latency_ms": round(float(r[4]), 1),
                "avg_quality": round(float(r[5]), 4),
            }
            for r in rows
        ]

    # Live: actual orchestra/routed cost + counterfactual locked-baseline cost per task.
    spec = baseline_spec()
    traffic = _live_ops_traffic_q(db).all()
    by_task_map: dict[str, dict] = {}
    for r in traffic:
        task = r.task_type or "unknown"
        bucket = by_task_map.setdefault(
            task,
            {
                "routed_cost": 0.0,
                "routed_n": 0,
                "routed_lat": 0.0,
                "routed_q": 0.0,
                "base_cost": 0.0,
                "base_n": 0,
            },
        )
        bucket["routed_cost"] += float(r.est_cost or 0)
        bucket["routed_n"] += 1
        bucket["routed_lat"] += float(r.latency_ms or 0)
        bucket["routed_q"] += float(r.quality_score or 0)
        bucket["base_cost"] += estimate_cost(
            spec, int(r.tokens_in or 0), int(r.tokens_out or 0)
        )
        bucket["base_n"] += 1

    out: list[dict] = []
    for task, b in by_task_map.items():
        n = b["routed_n"] or 1
        out.append(
            {
                "task_type": task,
                "policy": "routed",
                "requests": b["routed_n"],
                "total_cost": round(b["routed_cost"], 6),
                "avg_latency_ms": round(b["routed_lat"] / n, 1),
                "avg_quality": round(b["routed_q"] / n, 4),
            }
        )
        out.append(
            {
                "task_type": task,
                "policy": "baseline",
                "requests": b["base_n"],
                "total_cost": round(b["base_cost"], 6),
                "avg_latency_ms": round(float(spec.typical_latency_ms), 1),
                "avg_quality": round(b["routed_q"] / n, 4),
            }
        )
    out.sort(key=lambda row: (str(row["task_type"]), str(row["policy"])))
    return out


def eval_requests(db: Session, eval_run_id: str) -> list[dict]:
    rows = (
        db.query(RequestLog)
        .filter(RequestLog.eval_run_id == eval_run_id)
        .order_by(RequestLog.id.asc())
        .all()
    )
    return [
        {
            "id": r.id,
            "task_type": r.task_type,
            "policy": r.policy,
            "model": r.model,
            "provider": r.provider,
            "est_cost": r.est_cost,
            "latency_ms": r.latency_ms,
            "success": r.success,
            "quality_score": r.quality_score,
            "prompt": (r.prompt or "")[:120],
        }
        for r in rows
    ]


def eval_detail(db: Session, eval_run_id: str) -> dict:
    """Overview + matched task rollup + per-request rows for one eval run."""
    return {
        **overview(db, eval_run_id),
        "by_task": by_task(db, eval_run_id=eval_run_id),
        "requests": eval_requests(db, eval_run_id),
    }


def recent_requests(db: Session, limit: int = 50) -> list[dict]:
    rows = db.query(RequestLog).order_by(RequestLog.id.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "task_type": r.task_type,
            "policy": r.policy,
            "model": r.model,
            "provider": r.provider,
            "est_cost": r.est_cost,
            "latency_ms": r.latency_ms,
            "success": r.success,
            "fallback_used": r.fallback_used,
            "quality_score": r.quality_score,
            "prompt": r.prompt[:180],
            "response_preview": (r.response_preview or "")[:180],
            "eval_run_id": r.eval_run_id,
        }
        for r in rows
    ]


def latest_eval(db: Session) -> dict | None:
    run = db.query(EvalRun).order_by(EvalRun.created_at.desc()).first()
    if not run:
        return None
    return {
        "id": run.id,
        "created_at": run.created_at.isoformat(),
        "prompt_count": run.prompt_count,
        **eval_detail(db, run.id),
    }
