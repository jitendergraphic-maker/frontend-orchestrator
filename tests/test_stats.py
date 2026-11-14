"""Live Overview aggregation: orchestra counted; baseline = locked-model counterfactual."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, RequestLog
from app.registry import estimate_cost
from app.router import baseline_spec
from app.stats import by_model, by_task, overview


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
        quality_score=0.7,
    )
    defaults.update(kwargs)
    return RequestLog(**defaults)


def test_live_overview_counts_orchestra_and_excludes_mock_eval():
    db = _session()
    # Cheap actual est_cost, but enough tokens that locked baseline (fable) is expensive.
    db.add_all(
        [
            _row(
                policy="orchestra",
                est_cost=0.02,
                model="ca-claude-sonnet-5-5",
                tokens_in=20_000,
                tokens_out=10_000,
            ),
            _row(
                policy="routed",
                est_cost=0.03,
                model="ca-deepseek-v4-1-flash",
                tokens_in=20_000,
                tokens_out=10_000,
            ),
            _row(
                policy="baseline",
                est_cost=0.10,
                model="groq-gpt-oss-20b",
                provider="groq",
            ),
            _row(
                policy="baseline",
                est_cost=9.99,
                model="mock-strong",
                provider="mock",
            ),
            _row(
                policy="routed",
                est_cost=1.0,
                model="ca-deepseek-v4-1-flash",
                eval_run_id="eval-1",
            ),
            _row(
                policy="baseline",
                est_cost=2.0,
                model="groq-gpt-oss-20b",
                provider="groq",
                eval_run_id="eval-1",
            ),
        ]
    )
    db.commit()

    live = overview(db, scope="live")
    assert live["scope"] == "live"
    assert live["as_of"]
    assert live["routed"]["requests"] == 2
    assert abs(live["routed"]["total_cost"] - 0.05) < 1e-9
    # Baseline uses counterfactual on live orch+routed tokens (ignores cheap groq logs).
    assert live["baseline"]["requests"] == 2
    assert live["baseline"].get("cost_basis") == "counterfactual"
    spec = baseline_spec()
    expected = 2 * estimate_cost(spec, 20_000, 10_000)
    assert abs(live["baseline"]["total_cost"] - expected) < 1e-9
    assert live["baseline"]["total_cost"] > live["routed"]["total_cost"]
    assert live["cost_savings_vs_baseline"] is not None
    assert live["cost_savings_vs_baseline"] > 0

    all_scope = overview(db, scope="all")
    assert all_scope["scope"] == "all"
    # Legacy all: policy=routed only (not orchestra), includes eval routed row
    assert all_scope["routed"]["requests"] == 2  # live routed + eval routed
    assert abs(all_scope["routed"]["total_cost"] - 1.03) < 1e-9
    # baseline includes mock + live groq + eval groq
    assert all_scope["baseline"]["requests"] == 3
    assert abs(all_scope["baseline"]["total_cost"] - 12.09) < 1e-9


def test_live_by_model_and_by_task_exclude_mock_eval():
    db = _session()
    db.add_all(
        [
            _row(
                policy="orchestra",
                model="ca-claude-sonnet-5-5",
                est_cost=0.02,
                tokens_in=5_000,
                tokens_out=2_000,
            ),
            _row(
                policy="routed",
                model="mock-fast",
                provider="mock",
                est_cost=0.5,
            ),
            _row(
                policy="routed",
                model="ca-deepseek-v4-1-flash",
                est_cost=0.01,
                eval_run_id="eval-9",
            ),
        ]
    )
    db.commit()

    models = {r["model"]: r for r in by_model(db, scope="live")}
    assert models["ca-claude-sonnet-5-5"]["requests"] == 1
    assert models["ca-deepseek-v4-1-flash"]["requests"] == 0
    assert "mock-fast" not in models

    tasks = by_task(db, scope="live")
    # One task → routed row + counterfactual baseline row
    assert len(tasks) == 2
    by_pol = {t["policy"]: t for t in tasks}
    assert by_pol["routed"]["requests"] == 1
    assert by_pol["baseline"]["total_cost"] > by_pol["routed"]["total_cost"]

    all_tasks = by_task(db, scope="all")
    assert len(all_tasks) >= 2
