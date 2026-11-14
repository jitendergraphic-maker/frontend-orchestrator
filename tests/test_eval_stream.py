"""stream_eval yields started → step/step_done × N → final."""
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.eval_runner import stream_eval


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_stream_eval_event_sequence(monkeypatch, tmp_path):
    fixture = tmp_path / "prompts.json"
    fixture.write_text(
        '[{"prompt":"Write a loop","task_type":"code"},'
        '{"prompt":"Summarize this","task_type":"summarization"}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.eval_runner.get_settings",
        lambda: SimpleNamespace(eval_fixture_path=fixture),
    )

    calls: list[tuple[str, str | None]] = []

    def fake_complete(db, body, *, eval_run_id=None):
        calls.append((body.policy, body.task_type))
        return SimpleNamespace(
            task_type=body.task_type or "general",
            model="test-model",
            provider="test",
            latency_ms=12,
            est_cost=0.001,
            success=True,
        )

    monkeypatch.setattr("app.eval_runner.handle_complete", fake_complete)
    monkeypatch.setattr(
        "app.eval_runner.eval_detail",
        lambda db, run_id: {
            "routed": {"requests": 2},
            "baseline": {"requests": 2},
            "cost_savings_vs_baseline": 0.1,
            "scope": "eval",
            "as_of": "now",
            "by_task": [],
            "requests": [],
        },
    )

    db = _session()
    events = list(stream_eval(db, limit=2))

    names = [e[0] for e in events]
    assert names[0] == "started"
    assert names[-1] == "final"
    assert names.count("step") == 4
    assert names.count("step_done") == 4
    assert events[0][1]["prompt_count"] == 2
    assert events[0][1]["total_calls"] == 4

    steps = [e[1] for e in events if e[0] == "step"]
    assert [s["policy"] for s in steps] == ["baseline", "routed", "baseline", "routed"]
    assert steps[0]["status"].startswith("Calling baseline")
    assert steps[1]["status"].startswith("Routing prompt")

    final = events[-1][1]
    assert final["prompt_count"] == 2
    assert "id" in final
    assert final["routed"]["requests"] == 2
    assert calls == [
        ("baseline", "code"),
        ("routed", "code"),
        ("baseline", "summarization"),
        ("routed", "summarization"),
    ]
