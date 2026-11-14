from app.adapters.base import AdapterResult
from app.decomposer import decompose
from app.engine import handle_orchestra, stream_orchestra
from app.fallback import CompletionOutcome
from app.registry import load_models
from app.schemas import CompleteRequest
from app.synthesizer import stitch


class _FakeDB:
    def add(self, _row):
        return None

    def commit(self):
        return None


def test_decompose_mixed_prompt_into_specialist_parts():
    prompt = (
        "1) Write a Python function that reverses a string.\n\n"
        "2) Which number is larger, 9.11 or 9.8?\n\n"
        "3) Define TCP in one short paragraph.\n\n"
        "4) Write a four-line haiku about routing."
    )
    parts = decompose(prompt)
    assert len(parts) >= 3
    types = {p.task_type for p in parts}
    assert "code" in types
    assert "math" in types or "qa" in types
    assert "creative" in types or "qa" in types
    # Never invent text outside the original prompt
    joined = prompt
    for p in parts:
        assert p.text in joined or p.text.strip() in joined


def test_decompose_single_block_stays_one_part():
    parts = decompose("What is HTTP?")
    assert len(parts) == 1
    assert parts[0].task_type == "qa"


def test_decompose_blended_paragraph_without_numbers():
    prompt = (
        "I need a Python function that merges two sorted lists into one sorted "
        "list with a short doctest, then tell me which is larger between 0.999 "
        "and 0.99 with a brief place-value explanation, contrast REST and GraphQL "
        "for a mobile app that needs nested user and posts data in one short "
        "paragraph, compress all of that into three intern-friendly bullets, "
        "and finish with a four-line haiku about models handing work off to each other."
    )
    parts = decompose(prompt)
    assert len(parts) >= 4
    types = {p.task_type for p in parts}
    assert "code" in types
    assert "math" in types
    assert "creative" in types
    # Every slice must be substring of the original (no invented text)
    for p in parts:
        assert p.text in prompt or p.text.strip(" ,") in prompt


def test_stitch_labels_sections():
    from app.schemas import PartOutput, PromptPart

    parts = [
        PromptPart(id="p1", text="code bit", task_type="code", rationale="x"),
        PromptPart(id="p2", text="math bit", task_type="math", rationale="y"),
    ]
    outs = [
        PartOutput(
            part_id="p1",
            task_type="code",
            text="def f(): pass",
            model="m1",
            provider="groq",
            success=True,
        ),
        PartOutput(
            part_id="p2",
            task_type="math",
            text="9.11 is larger",
            model="m2",
            provider="gemini",
            success=True,
        ),
    ]
    text, method = stitch(parts, outs)
    assert method == "stitch"
    assert "## Code" in text and "## Math" in text
    assert "_model:" not in text
    assert "def f(): pass" in text
    assert "9.11 is larger" in text


def test_handle_orchestra_calls_each_part(monkeypatch):
    calls: list[str] = []

    def fake_complete(prompt, task_type, **kwargs):
        calls.append(task_type)
        models = load_models()
        spec = next(iter(models.values()))
        return CompletionOutcome(
            spec=spec,
            result=AdapterResult(
                output=f"answer for {task_type}",
                tokens_in=3,
                tokens_out=5,
                latency_ms=10,
            ),
            est_cost=0.0,
            fallback_used=False,
            fallback_from=None,
            ranked_ids=[spec.id],
            attempts=[],
            success=True,
        )

    monkeypatch.setattr("app.engine.complete_with_fallback", fake_complete)
    monkeypatch.setattr(
        "app.engine.rank_models",
        lambda task_type: [(next(iter(load_models().values())), 1.0)],
    )
    monkeypatch.setattr(
        "app.engine._scoreboard_for",
        lambda task_type: [],
    )

    body = CompleteRequest(
        prompt=(
            "1) Write a Python function that adds two numbers.\n\n"
            "2) Calculate 2+2.\n\n"
            "3) Write a four-line haiku about clouds."
        ),
        policy="orchestra",
    )
    resp = handle_orchestra(_FakeDB(), body)
    assert resp.policy == "orchestra"
    assert resp.synthesis_method == "stitch"
    assert len(resp.parts) >= 2
    assert len(resp.part_outputs) == len(resp.parts)
    assert len(calls) == len(resp.parts)
    assert resp.success is True
    assert "Combined from" in resp.output


def test_stream_orchestra_emits_decompose_and_synthesize(monkeypatch):
    models = load_models()
    spec = next(iter(models.values()))

    def fake_complete(prompt, task_type, **kwargs):
        on_event = kwargs.get("on_event")
        if on_event:
            on_event(
                "attempt_start",
                {"index": 0, "model": spec.id, "provider": spec.provider},
            )
            on_event(
                "attempt_end",
                {
                    "index": 0,
                    "model": spec.id,
                    "provider": spec.provider,
                    "outcome": "success",
                    "reason": None,
                    "latency_ms": 5,
                    "tokens_in": 1,
                    "tokens_out": 2,
                    "est_cost": 0.0,
                },
            )
        return CompletionOutcome(
            spec=spec,
            result=AdapterResult("ok", 1, 2, 5),
            est_cost=0.0,
            fallback_used=False,
            fallback_from=None,
            ranked_ids=[spec.id],
            attempts=[],
            success=True,
        )

    monkeypatch.setattr("app.engine.complete_with_fallback", fake_complete)
    monkeypatch.setattr(
        "app.engine.rank_models",
        lambda task_type: [(spec, 1.0)],
    )
    monkeypatch.setattr("app.engine._scoreboard_for", lambda task_type: [])

    body = CompleteRequest(
        prompt="1) Write a Python function.\n\n2) Calculate 3*3.",
        policy="orchestra",
    )
    events = []
    gen = stream_orchestra(_FakeDB(), body)
    try:
        while True:
            events.append(next(gen))
    except StopIteration:
        pass

    names = [e[0] for e in events]
    assert "decomposed" in names
    assert "part_ranked" in names
    assert "part_done" in names
    assert "synthesized" in names
    assert "final" in names
    assert names.index("decomposed") < names.index("synthesized")
    assert names.index("synthesized") < names.index("final")
    # All rankings emitted before any specialist finishes.
    assert max(i for i, n in enumerate(names) if n == "part_ranked") < names.index(
        "part_done"
    )


def test_stream_orchestra_runs_parts_in_parallel(monkeypatch):
    import threading
    import time

    models = load_models()
    spec = next(iter(models.values()))
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_complete(prompt, task_type, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.12)
        with lock:
            active -= 1
        return CompletionOutcome(
            spec=spec,
            result=AdapterResult(f"answer-{task_type}", 1, 2, 120),
            est_cost=0.0,
            fallback_used=False,
            fallback_from=None,
            ranked_ids=[spec.id],
            attempts=[],
            success=True,
        )

    monkeypatch.setattr("app.engine.complete_with_fallback", fake_complete)
    monkeypatch.setattr(
        "app.engine.rank_models",
        lambda task_type: [(spec, 1.0)],
    )
    monkeypatch.setattr("app.engine._scoreboard_for", lambda task_type: [])

    body = CompleteRequest(
        prompt=(
            "1) Write a Python function that adds.\n\n"
            "2) Calculate 2+2.\n\n"
            "3) Write a four-line haiku about clouds."
        ),
        policy="orchestra",
    )
    started = time.perf_counter()
    events = list(stream_orchestra(_FakeDB(), body))
    elapsed = time.perf_counter() - started

    part_dones = [e for e in events if e[0] == "part_done"]
    assert len(part_dones) >= 2
    # Parallel: wall time closer to one sleep than N sleeps.
    assert elapsed < 0.12 * len(part_dones) * 0.85
    assert max_active >= 2


def test_classify_math():
    from app.classifier import classify

    assert classify("Calculate the derivative of x^2") == "math"


def test_diversify_ranked_prefers_unused_provider():
    from app.registry import ModelSpec
    from app.router import diversify_ranked

    def spec(id_: str, provider: str) -> ModelSpec:
        return ModelSpec(
            id=id_,
            provider=provider,
            display_name=id_,
            api_model=id_,
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
            typical_latency_ms=500,
            strengths={"qa": 0.9},
        )

    a = spec("gemini-a", "gemini")
    b = spec("groq-b", "groq")
    c = spec("gemini-c", "gemini")
    ranked = [(a, 0.47), (b, 0.45), (c, 0.44)]

    # No used providers → keep score order
    assert [m.id for m in diversify_ranked(ranked, set(), margin=0.08)] == [
        "gemini-a",
        "groq-b",
        "gemini-c",
    ]
    # Gemini already used → promote groq within margin
    out = diversify_ranked(ranked, {"gemini"}, margin=0.08)
    assert out[0].id == "groq-b"
    assert out[0].provider == "groq"
    # Outside margin → keep top
    out2 = diversify_ranked(ranked, {"gemini"}, margin=0.01)
    assert out2[0].id == "gemini-a"
