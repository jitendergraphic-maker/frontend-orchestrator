"""Deterministic stitch of specialist outputs (no extra LLM judge)."""
from __future__ import annotations

import re

from app.schemas import PartOutput, PromptPart, SynthesisMethod

_TASK_TITLES = {
    "qa": "QA",
    "code": "Code",
    "debug": "Debug",
    "math": "Math",
    "reasoning": "Reasoning",
    "summarization": "Summary",
    "creative": "Creative",
    "translation": "Translation",
    "extraction": "Extraction",
    "analysis": "Analysis",
    "chat": "Chat",
    "vision": "Vision",
}

# Models sometimes invent meta-refusals when they saw other slices in context.
_REFUSAL_LINE = re.compile(
    r"(?i)^\s*(?:"
    r"this section is for\b.+\bnot part of\b.+"
    r"|.*\bnot part of (?:the |this |your )?(?:qa |assigned )?slice\b.*"
    r"|.*\bnot part of (?:this |your )?assignment\b.*"
    r"|.*\bhandled (?:separately|by another specialist)\b.*"
    r")\s*$"
)


def clean_specialist_text(text: str) -> str:
    """Drop slice-refusal boilerplate; keep real answer lines."""
    if not text:
        return ""
    kept: list[str] = []
    for line in text.splitlines():
        if _REFUSAL_LINE.match(line.strip()):
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    # Collapse leftover empty runs after stripping.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def stitch(
    parts: list[PromptPart],
    outputs: list[PartOutput],
) -> tuple[str, SynthesisMethod]:
    by_id = {o.part_id: o for o in outputs}
    sections: list[str] = []
    n_ok = sum(1 for o in outputs if o.success)
    sections.append(f"Combined from {n_ok}/{len(outputs)} specialist(s).")
    for part in parts:
        out = by_id.get(part.id)
        title = _TASK_TITLES.get(part.task_type, part.task_type.title())
        header = f"## {title}"
        if out is None:
            sections.append(f"{header}\n[missing specialist output]")
            continue
        raw = (out.text or "").strip()
        body = clean_specialist_text(raw) if raw else ""
        if not body:
            body = (
                f"[specialist failed: {out.error or 'unknown'}]"
                if not out.success
                else "[empty specialist output]"
            )
        sections.append(f"{header}\n\n{body}")
    return "\n\n".join(sections), "stitch"
