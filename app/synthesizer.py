"""Deterministic stitch of specialist outputs (no extra LLM judge)."""
from __future__ import annotations

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
        body = (out.text or "").strip() or (
            f"[specialist failed: {out.error or 'unknown'}]"
        )
        sections.append(f"{header}\n\n{body}")
    return "\n\n".join(sections), "stitch"
