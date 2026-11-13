from app.classifier import classify
from app.adapters.base import AdapterResult
from app.adapters.groq import GroqAdapter
from app.adapters.nvidia import NvidiaAdapter
from app.fallback import complete_with_fallback
from app.quality import quality_score
from app.registry import load_models
from app.router import provider_available, rank_models, rank_models_detailed


def test_classifier_hint_and_rules():
    assert classify("hello", "qa") == "qa"
    assert classify("Write a Python function that sorts a list") == "code"
    assert classify("TL;DR this paragraph about databases") == "summarization"
    assert (
        classify("Think step by step: why is fallback cheaper on average?")
        == "reasoning"
    )
    assert classify("Write a haiku about logs") == "creative"
    assert classify("What is HTTP?") == "qa"


def test_rank_uses_only_configured_real_models(monkeypatch):
    monkeypatch.setattr(
        "app.router.provider_status", lambda spec: (True, None)
    )
    ranked = [spec.id for spec, _ in rank_models("code")]
    assert set(ranked) == {
        "groq-gpt-oss-20b",
        "groq-gpt-oss-120b",
        "groq-qwen3-27b",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "nvidia-llama-3.2-11b",
        "nvidia-glm-5.3-flash",
        "or-lfm-2.5-free",
        "or-nemotron-lightning-free",
        "or-auto-free",
        "or-laguna-xs-free",
        "or-north-mini-code-free",
        "or-laguna-s-free",
    }


def test_groq_requires_api_key(monkeypatch):
    settings = type("Settings", (), {"groq_api_key": ""})()
    monkeypatch.setattr("app.router.get_settings", lambda: settings)
    groq_model = load_models()["groq-gpt-oss-20b"]

    assert not provider_available(groq_model)
    settings.groq_api_key = "configured"
    assert provider_available(groq_model)


def test_nvidia_requires_api_key(monkeypatch):
    settings = type("Settings", (), {"nvidia_api_key": ""})()
    monkeypatch.setattr("app.router.get_settings", lambda: settings)
    nvidia_model = load_models()["nvidia-llama-3.2-11b"]

    assert not provider_available(nvidia_model)
    settings.nvidia_api_key = "configured"
    assert provider_available(nvidia_model)


def test_gemini_requires_api_key(monkeypatch):
    settings = type("Settings", (), {"gemini_api_key": ""})()
    monkeypatch.setattr("app.router.get_settings", lambda: settings)
    gemini_model = load_models()["gemini-3.5-flash-lite"]

    assert not provider_available(gemini_model)
    settings.gemini_api_key = "configured"
    assert provider_available(gemini_model)


def test_nvidia_adapter_uses_openai_compatible_endpoint(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "A concise answer"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }

    request = {}

    def post(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(
        "app.adapters.nvidia.get_settings",
        lambda: type(
            "Settings", (), {"nvidia_api_key": "test-key", "completion_max_tokens": 1536}
        )(),
    )
    monkeypatch.setattr("app.adapters.nvidia.httpx.post", post)

    result = NvidiaAdapter().complete(
        api_model="meta/llama-3.2-11b-vision-instruct", prompt="Question", timeout=5
    )

    assert result.output == "A concise answer"
    assert request["url"] == (
        "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["json"]["model"] == "meta/llama-3.2-11b-vision-instruct"


def test_quality_empty_is_zero():
    assert quality_score("q", "", "qa", 0.9) == 0.0
    assert quality_score("q", "```python\npass\n```", "code", 0.9) > 0.5


def test_fallback_continues_after_unexpected_adapter_error(monkeypatch):
    calls = []

    def call(spec, prompt):
        calls.append(spec.id)
        if len(calls) == 1:
            raise RuntimeError("provider response could not be parsed")
        return AdapterResult("usable answer", 8, 4, 25)

    monkeypatch.setattr("app.fallback._call", call)
    models = load_models()
    outcome = complete_with_fallback(
        "What is HTTP?",
        "qa",
        ranked=[models["groq-gpt-oss-20b"], models["nvidia-llama-3.2-11b"]],
    )

    assert calls == ["groq-gpt-oss-20b", "nvidia-llama-3.2-11b"]
    assert outcome.spec.id == "nvidia-llama-3.2-11b"
    assert outcome.fallback_used is True


def test_groq_retries_rate_limit_once(monkeypatch):
    class Response:
        def __init__(self, status_code):
            self.status_code = status_code
            self.text = "rate limited"

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "def reverse(value): return value[::-1]"
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 12},
            }

    responses = iter([Response(429), Response(200)])
    requests = []
    urls = []
    sleeps = []

    def post(*args, **kwargs):
        urls.append(args[0])
        requests.append(kwargs)
        return next(responses)

    monkeypatch.setattr(
        "app.adapters.groq.get_settings",
        lambda: type(
            "Settings", (), {"groq_api_key": "test-key", "completion_max_tokens": 1536}
        )(),
    )
    monkeypatch.setattr("app.adapters.groq.httpx.post", post)
    monkeypatch.setattr("app.adapters.groq.time.sleep", sleeps.append)

    result = GroqAdapter().complete(
        api_model="openai/gpt-oss-20b", prompt="Write reverse", timeout=5
    )

    assert result.output.startswith("def reverse")
    assert len(requests) == 2
    assert sleeps == [2]
    assert urls == [
        "https://api.groq.com/openai/v1/chat/completions",
        "https://api.groq.com/openai/v1/chat/completions",
    ]
    assert requests[0]["headers"]["Authorization"] == "Bearer test-key"
    assert requests[0]["json"]["max_tokens"] == 1536


def test_strict_baseline_never_calls_second_model(monkeypatch):
    calls = []

    def call(spec, prompt):
        calls.append(spec.id)
        raise RuntimeError("baseline provider down")

    monkeypatch.setattr("app.fallback._call", call)
    models = load_models()
    outcome = complete_with_fallback(
        "What is HTTP?",
        "qa",
        ranked=[models["groq-gpt-oss-20b"], models["nvidia-llama-3.2-11b"]],
        allow_fallback=False,
    )

    assert calls == ["groq-gpt-oss-20b"]
    assert outcome.success is False
    assert outcome.fallback_used is False
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].outcome == "error"


def test_attempt_trace_records_fallback_chain(monkeypatch):
    def call(spec, prompt):
        if spec.id == "groq-gpt-oss-20b":
            raise RuntimeError("timeout")
        return AdapterResult("ok", 4, 2, 18)

    monkeypatch.setattr("app.fallback._call", call)
    models = load_models()
    outcome = complete_with_fallback(
        "What is HTTP?",
        "qa",
        ranked=[models["groq-gpt-oss-20b"], models["nvidia-llama-3.2-11b"]],
    )

    assert [a.outcome for a in outcome.attempts] == ["error", "success"]
    assert outcome.attempts[0].reason == "timeout"
    assert outcome.attempts[1].model == "nvidia-llama-3.2-11b"
    assert outcome.fallback_used is True


def test_scoreboard_finite_and_deterministic(monkeypatch):
    monkeypatch.setattr(
        "app.router.provider_status", lambda spec: (True, None)
    )
    a = rank_models_detailed("code")
    b = rank_models_detailed("code")
    assert [row.id for row in a] == [row.id for row in b]
    for row in a:
        assert row.eligible
        assert row.score == row.score  # not NaN
        assert abs(row.score) != float("inf")
        assert 0.0 <= row.cost_norm <= 1.0
        assert 0.0 <= row.latency_norm <= 1.0


def test_fallback_attempt_trace_on_success_path(monkeypatch):
    monkeypatch.setattr(
        "app.fallback._call",
        lambda spec, prompt: AdapterResult("hello world answer", 3, 5, 12),
    )
    models = load_models()
    outcome = complete_with_fallback(
        "Hi",
        "qa",
        ranked=[models["nvidia-llama-3.2-11b"]],
    )
    assert outcome.success is True
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].outcome == "success"
    assert outcome.attempts[0].est_cost >= 0.0
