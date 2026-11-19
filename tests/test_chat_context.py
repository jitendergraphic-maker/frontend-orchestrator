from app.chat_context import (
    MAX_TURNS,
    adapter_messages,
    decompose_input,
    resolve_prompt,
    trim_messages,
)
from app.schemas import ChatTurn, CompleteRequest


def test_resolve_prompt_from_messages():
    body = CompleteRequest(
        prompt="",
        messages=[
            ChatTurn(role="user", content="First ask"),
            ChatTurn(role="assistant", content="First answer"),
            ChatTurn(role="user", content="Make it shorter"),
        ],
        policy="routed",
    )
    assert resolve_prompt(body) == "Make it shorter"


def test_resolve_prompt_falls_back_to_prompt_field():
    body = CompleteRequest(prompt="Plain prompt", policy="baseline")
    assert resolve_prompt(body) == "Plain prompt"


def test_decompose_input_includes_prior_user_turns():
    body = CompleteRequest(
        prompt="Make it shorter",
        messages=[
            ChatTurn(role="user", content="Write a long essay about cats"),
            ChatTurn(role="assistant", content="Once upon a time…"),
            ChatTurn(role="user", content="Make it shorter"),
        ],
    )
    text = decompose_input(body)
    assert "Write a long essay about cats" in text
    assert "Make it shorter" in text


def test_trim_messages_caps_turns():
    turns = [ChatTurn(role="user", content=f"u{i}") for i in range(20)]
    trimmed = trim_messages(turns)
    assert len(trimmed) == MAX_TURNS
    assert trimmed[0].content == "u8"
    assert trimmed[-1].content == "u19"


def test_adapter_messages_preserves_history():
    body = CompleteRequest(
        prompt="Follow up",
        messages=[
            ChatTurn(role="user", content="Hello"),
            ChatTurn(role="assistant", content="Hi there"),
            ChatTurn(role="user", content="Follow up"),
        ],
    )
    msgs = adapter_messages(body)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[-1]["content"] == "Follow up"


def test_adapter_messages_override_for_specialist():
    body = CompleteRequest(
        prompt="Do both",
        messages=[
            ChatTurn(role="user", content="Earlier"),
            ChatTurn(role="assistant", content="Ok"),
            ChatTurn(role="user", content="Do both"),
        ],
    )
    msgs = adapter_messages(body, override_user="Specialist: write code only")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Specialist: write code only"
    assert "Earlier" not in msgs[0]["content"]
    assert "Prior conversation" not in msgs[0]["content"]


def test_complete_request_requires_user_in_messages():
    try:
        CompleteRequest(
            prompt="",
            messages=[ChatTurn(role="assistant", content="only assistant")],
        )
        assert False, "expected validation error"
    except Exception:
        pass


def test_mock_adapter_accepts_messages():
    from app.adapters.mock import MockAdapter

    result = MockAdapter().complete(
        api_model="mock-fast",
        prompt="ignored",
        timeout=5,
        messages=[
            {"role": "user", "content": "What is HTTP?"},
            {"role": "assistant", "content": "A protocol."},
            {"role": "user", "content": "Say it in one word"},
        ],
    )
    assert result.output
    assert "[context: 3 turns]" in result.output
