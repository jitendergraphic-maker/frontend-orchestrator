"""One-shot / empty-DB import of fixtures/db_seed.json into the active DATABASE_URL."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select

from app.config import get_settings
from app.db import EvalRun, RequestLog, SessionLocal

log = logging.getLogger("seed_import")
SEED_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "db_seed.json"


def _parse_dt(value):
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def import_seed_if_needed() -> dict:
    settings = get_settings()
    force = bool(getattr(settings, "import_local_seed", False))
    if not SEED_PATH.exists():
        return {"skipped": True, "reason": "no_seed_file"}

    db = SessionLocal()
    try:
        existing_logs = db.scalar(select(func.count()).select_from(RequestLog)) or 0
        if existing_logs > 0 and not force:
            return {"skipped": True, "reason": "db_not_empty", "request_logs": existing_logs}

        payload = json.loads(SEED_PATH.read_text(encoding="utf-8"))
        eval_rows = payload.get("eval_runs") or []
        log_rows = payload.get("request_logs") or []

        # Force sync = replace hosted rows with the exported local snapshot.
        if force and existing_logs > 0:
            db.query(RequestLog).delete()
            db.query(EvalRun).delete()
            db.commit()

        eval_added = 0
        for row in eval_rows:
            rid = row.get("id")
            if not rid:
                continue
            if db.get(EvalRun, rid):
                continue
            db.add(
                EvalRun(
                    id=rid,
                    created_at=_parse_dt(row.get("created_at")),
                    prompt_count=int(row.get("prompt_count") or 0),
                    notes=row.get("notes"),
                )
            )
            eval_added += 1

        log_added = 0
        for row in log_rows:
            # Drop local autoincrement id so hosted DB assigns its own (avoids PK clashes).
            db.add(
                RequestLog(
                    created_at=_parse_dt(row.get("created_at")),
                    prompt=str(row.get("prompt") or ""),
                    task_type=str(row.get("task_type") or "qa"),
                    policy=str(row.get("policy") or "routed"),
                    model=str(row.get("model") or "unknown"),
                    provider=str(row.get("provider") or "unknown"),
                    tokens_in=int(row.get("tokens_in") or 0),
                    tokens_out=int(row.get("tokens_out") or 0),
                    est_cost=float(row.get("est_cost") or 0.0),
                    latency_ms=int(row.get("latency_ms") or 0),
                    success=bool(row.get("success", True)),
                    fallback_used=bool(row.get("fallback_used", False)),
                    fallback_from=row.get("fallback_from"),
                    quality_score=float(row.get("quality_score") or 0.0),
                    error=row.get("error"),
                    response_preview=row.get("response_preview"),
                    eval_run_id=row.get("eval_run_id"),
                )
            )
            log_added += 1

        db.commit()
        result = {
            "skipped": False,
            "eval_runs_added": eval_added,
            "request_logs_added": log_added,
            "forced": force,
        }
        log.info("seed import complete: %s", result)
        return result
    except Exception:
        db.rollback()
        log.exception("seed import failed")
        raise
    finally:
        db.close()
