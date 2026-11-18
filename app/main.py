import json

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db, init_db
from app.engine import handle_complete, stream_complete
from app.eval_runner import run_eval, stream_eval
from app.registry import load_models, load_routing
from app.router import rank_models_detailed
from app.schemas import CompleteRequest, EvalRunRequest
from app.seed_import import import_seed_if_needed
from app.stats import by_model, by_task, latest_eval, overview, recent_requests

app = FastAPI(title="LLM Orchestration Engine", version="0.2.0")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()
    # Loads fixtures/db_seed.json when DB is empty (or IMPORT_LOCAL_SEED=true).
    import_seed_if_needed()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/complete")
def complete(body: CompleteRequest, db: Session = Depends(get_db)):
    return handle_complete(db, body)


@app.post("/v1/complete/stream")
def complete_stream(body: CompleteRequest, db: Session = Depends(get_db)):
    """NDJSON stream of routing theater events, ending with event=final."""

    def generate():
        for event, payload in stream_complete(db, body):
            line = json.dumps({"event": event, "data": payload}, default=str)
            yield line + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1/models")
def models():
    routing = load_routing()
    items = []
    for spec in load_models().values():
        items.append(
            {
                "id": spec.id,
                "provider": spec.provider,
                "display_name": spec.display_name,
                "input_cost_per_1k": spec.input_cost_per_1k,
                "output_cost_per_1k": spec.output_cost_per_1k,
                "billable": spec.billable,
                "typical_latency_ms": spec.typical_latency_ms,
                "strengths": spec.strengths,
                "enabled": spec.enabled,
            }
        )
    return {
        "baseline_model": routing.baseline_model,
        "weights": {
            "quality": routing.quality,
            "cost": routing.cost,
            "latency": routing.latency,
        },
        "models": items,
        "quality_label": "heuristic quality proxy",
    }


@app.get("/v1/rank")
def rank(task_type: str = Query(default="qa")):
    detailed = rank_models_detailed(task_type, include_unavailable=True)
    return {
        "task_type": task_type,
        "ranked": [row.model_dump() for row in detailed if row.eligible],
        "scoreboard": [row.model_dump() for row in detailed],
        "formula": "score = w_q*strength - w_c*cost_norm - w_l*latency_norm",
    }


@app.get("/v1/stats/overview")
def stats_overview(
    eval_run_id: str | None = None,
    scope: str = "live",
    db: Session = Depends(get_db),
):
    return overview(db, eval_run_id, scope=scope)


@app.get("/v1/stats/by-model")
def stats_models(
    include_retired: bool = False,
    scope: str = "live",
    db: Session = Depends(get_db),
):
    return by_model(db, include_retired=include_retired, scope=scope)


@app.get("/v1/stats/by-task")
def stats_tasks(
    scope: str = "live",
    eval_run_id: str | None = None,
    db: Session = Depends(get_db),
):
    return by_task(db, scope=scope, eval_run_id=eval_run_id)


@app.get("/v1/requests")
def list_requests(limit: int = 50, db: Session = Depends(get_db)):
    return recent_requests(db, limit)


@app.post("/v1/eval/run")
def eval_run(body: EvalRunRequest | None = None, db: Session = Depends(get_db)):
    body = body or EvalRunRequest()
    return run_eval(db, limit=body.limit, notes=body.notes)


@app.post("/v1/eval/run/stream")
def eval_run_stream(body: EvalRunRequest | None = None, db: Session = Depends(get_db)):
    """NDJSON stream of eval progress events, ending with event=final."""
    body = body or EvalRunRequest()

    def generate():
        for event, payload in stream_eval(db, limit=body.limit, notes=body.notes):
            yield json.dumps({"event": event, "data": payload}, default=str) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1/eval/latest")
def eval_latest(db: Session = Depends(get_db)):
    return latest_eval(db) or {"id": None}
