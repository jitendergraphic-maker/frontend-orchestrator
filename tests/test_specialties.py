"""Specialty priors: with all providers eligible, task winners should match roles."""
from app.classifier import classify
from app.decomposer import classify_segment, decompose
from app.registry import load_models, load_routing
from app.router import rank_models


def _top_ids(task_type: str, n: int = 3) -> list[str]:
    return [spec.id for spec, _ in rank_models(task_type)[:n]]


def test_routing_weights_favor_quality():
    load_routing.cache_clear()
    routing = load_routing()
    assert routing.quality == 0.65
    assert routing.cost == 0.20
    assert routing.latency == 0.15
    assert routing.diversity_score_margin == 0.12


def test_all_models_have_twelve_strength_axes():
    load_models.cache_clear()
    expected = {
        "qa",
        "code",
        "debug",
        "math",
        "reasoning",
        "summarization",
        "creative",
        "translation",
        "extraction",
        "analysis",
        "chat",
        "vision",
    }
    for spec in load_models().values():
        assert expected <= set(spec.strengths.keys()), spec.id


def test_specialty_winners_in_top3(monkeypatch):
    monkeypatch.setattr("app.router.provider_status", lambda spec: (True, None))
    load_models.cache_clear()
    load_routing.cache_clear()

    assert any(m in _top_ids("code") for m in ("cf-qwen-coder-32b", "cf-kimi-code", "or-north-mini-code-free"))
    assert any(m in _top_ids("debug") for m in ("cf-qwen-coder-32b", "cf-kimi-code"))
    assert any(m in _top_ids("math") for m in ("cf-qwen3.8-27b", "cf-glm-5.3-flash", "cf-deepseek-r1"))
    assert any(m in _top_ids("reasoning") for m in ("cf-deepseek-r1", "cf-qwq-32b"))
    assert any(m in _top_ids("qa") for m in ("cf-llama-3.3-70b", "ca-gemini-3-8-flash", "cf-gpt-oss-120b"))
    assert any(
        m in _top_ids("summarization")
        for m in ("gemini-3.5-flash-lite", "cf-llama-3.3-70b", "ca-gemini-3-8-flash")
    )
    assert any(m in _top_ids("creative") for m in ("ca-gpt-6-luna", "gemini-3.5-flash-lite", "cf-llama-3.3-70b"))
    assert any(m in _top_ids("translation") for m in ("co-command-a-translate", "co-north-small-translate", "cf-qwen3.8-27b"))
    assert any(m in _top_ids("extraction") for m in ("cf-mistral-small", "cf-qwen-coder-32b"))
    assert any(m in _top_ids("analysis") for m in ("cf-mistral-small", "cf-glm-5.3-flash"))
    assert any(m in _top_ids("chat") for m in ("cf-llama-3.1-8b", "cf-gpt-oss-20b", "ca-gpt-6-luna"))
    assert "cf-llama-vision" in _top_ids("vision", 1)

    # Core specialties should stay on Cloudflare / Cohere specialists
    for task in ("code", "debug", "math", "reasoning", "extraction", "analysis"):
        assert _top_ids(task, 1)[0].startswith("cf-"), task
    assert _top_ids("translation", 1)[0].startswith("co-")
    assert _top_ids("vision", 1)[0].startswith(("cf-", "co-"))


def test_code_specialist_outranks_generalist(monkeypatch):
    monkeypatch.setattr("app.router.provider_status", lambda spec: (True, None))
    load_models.cache_clear()
    load_routing.cache_clear()
    ranked = [spec.id for spec, _ in rank_models("code")]
    assert ranked.index("cf-qwen-coder-32b") < ranked.index("gemini-3.5-flash-lite")
    assert ranked.index("cf-qwen-coder-32b") < ranked.index("groq-qwen3-27b")


def test_classify_new_task_types():
    assert classify("Fix this TypeError traceback in my Python script") == "debug"
    assert classify("Translate this sentence to Spanish") == "translation"
    assert classify("Extract the names as a JSON object") == "extraction"
    assert classify("Compare REST vs GraphQL with pros and cons") == "analysis"
    assert classify("Hey there, how are you?") == "chat"
    assert classify("Describe this screenshot of the dashboard") == "vision"
    assert classify_segment("debug the failing test")[0] == "debug"


def test_decompose_recognizes_new_intents_in_blend():
    prompt = (
        "Write a Python function that adds two numbers, then fix this TypeError: x=None, "
        "translate hello to French, extract the emails as JSON, analyze pros and cons of SQLite, "
        "and finish with a four-line haiku about routing."
    )
    parts = decompose(prompt)
    types = {p.task_type for p in parts}
    assert len(parts) >= 4
    assert "code" in types or "debug" in types
    assert "translation" in types or "extraction" in types or "analysis" in types
    assert "creative" in types
