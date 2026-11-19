"""Multi-turn chat helpers for CompleteRequest.messages[]."""

from __future__ import annotations

from typing import Any

from app.schemas import ChatTurn, CompleteRequest

MAX_TURNS = 12
MAX_CHARS = 8000


def trim_messages(messages: list[ChatTurn]) -> list[ChatTurn]:
    """Keep the newest turns within turn and character budgets."""
    if not messages:
        return []
    trimmed = list(messages[-MAX_TURNS:])
    while trimmed and sum(len(m.content) for m in trimmed) > MAX_CHARS:
        trimmed = trimmed[1:]
    return trimmed


def resolve_prompt(body: CompleteRequest) -> str:
    """Latest user text — from messages if present, else prompt."""
    if body.messages:
        for turn in reversed(body.messages):
            if turn.role == "user" and turn.content.strip():
                return turn.content.strip()
    return (body.prompt or "").strip()


def decompose_input(body: CompleteRequest) -> str:
    """Text used for classify/decompose: latest user turn, plus prior user turns for pronouns."""
    prompt = resolve_prompt(body)
    if not body.messages:
        return prompt
    prior_users = [
        m.content.strip()
        for m in trim_messages(body.messages)[:-1]
        if m.role == "user" and m.content.strip()
    ]
    if not prior_users:
        return prompt
    # Last 1–2 prior user turns give enough context for short follow-ups.
    ctx = "\n".join(prior_users[-2:])
    return f"{ctx}\n\n{prompt}"


def history_blurb(body: CompleteRequest, *, max_chars: int = 600) -> str:
    """Short prior-conversation summary for orchestra specialist prompts."""
    if not body.messages:
        return ""
    prompt = resolve_prompt(body)
    prior = [
        m
        for m in trim_messages(body.messages)
        if not (m.role == "user" and m.content.strip() == prompt)
    ]
    if not prior:
        # Drop the trailing current user turn if it matches prompt.
        turns = trim_messages(body.messages)
        if turns and turns[-1].role == "user":
            prior = turns[:-1]
        else:
            prior = turns
    if not prior:
        return ""
    lines: list[str] = []
    budget = max_chars
    for m in prior:
        label = "User" if m.role == "user" else "Assistant" if m.role == "assistant" else "System"
        snippet = m.content.strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "…"
        line = f"{label}: {snippet}"
        if len(line) > budget:
            break
        lines.append(line)
        budget -= len(line) + 1
    return "\n".join(lines)


def adapter_messages(
    body: CompleteRequest,
    *,
    override_user: str | None = None,
) -> list[dict[str, str]]:
    """
    Build OpenAI-style messages for adapters.

    If override_user is set (orchestra specialist), return ONLY that text —
    never re-attach prior conversation (it re-injects multi-section prompts and
    causes "handled separately" walkthrough junk).
    """
    prompt = resolve_prompt(body)
    if override_user is not None:
        return [{"role": "user", "content": override_user}]

    if not body.messages:
        return [{"role": "user", "content": prompt}]

    turns = trim_messages(body.messages)
    out: list[dict[str, str]] = []
    for m in turns:
        role = m.role if m.role in ("user", "assistant", "system") else "user"
        out.append({"role": role, "content": m.content})

    if not out or out[-1]["role"] != "user":
        if prompt:
            out.append({"role": "user", "content": prompt})
    elif prompt and out[-1]["content"].strip() != prompt:
        out.append({"role": "user", "content": prompt})

    return out if out else [{"role": "user", "content": prompt or " "}]


def estimate_prompt_chars(messages: list[dict[str, Any]] | None, prompt: str) -> int:
    if messages:
        return max(1, sum(len(str(m.get("content") or "")) for m in messages))
    return max(1, len(prompt))
