"""Eval-scoped by_task / eval_detail for Baseline charts."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, EvalRun, RequestLog
from app.stats import by_task, eval_detail, eval_requests


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _row(**kwargs) -> RequestLog:
    defaults = dict(
        prompt="p",
        task_type="code",
        policy="routed",
        model="ca-deepseek-v4-1-flash",
        provider="coarena",
        tokens_in=10,
        tokens_out=5,
        est_cost=0.01,
        latency_ms=100,
        success=True,
        fallback_used=False,
        quality_score=0.8,
    )
    defaults.update(kwargs)
    return RequestLog(**defaults)


def test_by_task_scoped_to_eval_run():
    db = _session()
    db.add(EvalRun(id="eval-a", prompt_count=2))
    db.add_all(
        [
            _row(eval_run_id="eval-a", policy="baseline", task_type="code", est_cost=1.0),
            _row(eval_run_id="eval-a", policy="routed", task_type="code", est_cost=0.05),
            _row(
                eval_run_id="eval-a",
                policy="baseline",
                task_type="qa",
                est_cost=0.8,
                model="cf-gpt-oss-120b",
            ),
            _row(eval_run_id="eval-a", policy="routed", task_type="qa", est_cost=0.02),
            # Other run / live noise must not appear
            _row(eval_run_id="eval-b", policy="baseline", task_type="code", est_cost=9.0),
            _row(policy="orchestra", task_type="code", est_cost=0.5),
        ]
    )
    db.commit()

    rows = by_task(db, eval_run_id="eval-a")
    assert len(rows) == 4
    by_key = {(r["task_type"], r["policy"]): r for r in rows}
    assert by_key[("code", "baseline")]["total_cost"] == 1.0
    assert by_key[("code", "routed")]["total_cost"] == 0.05
    assert by_key[("qa", "baseline")]["requests"] == 1

    detail = eval_detail(db, "eval-a")
    assert detail["scope"] == "eval"
    assert detail["baseline"]["requests"] == 2
    assert detail["routed"]["requests"] == 2
    assert abs(detail["baseline"]["total_cost"] - 1.8) < 1e-9
    assert abs(detail["routed"]["total_cost"] - 0.07) < 1e-9
    assert detail["cost_savings_vs_baseline"] is not None
    assert len(detail["by_task"]) == 4
    assert len(eval_requests(db, "eval-a")) == 4
