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

# Meta-refusal / walkthrough lines free models invent for other slices.
_REFUSAL_LINE = re.compile(
    r"(?i)^\s*(?:"
    r".*\bthis section is for\b.*"
    r"|.*\bwill be handled separately\b.*"
    r"|.*\bhandled separately\b.*"
    r"|.*\bhandled by another(?: specialist)?\b.*"
    r"|.*\bnot part of (?:the |this |your )?(?:qa |assigned )?slice\b.*"
    r"|.*\bnot part of (?:this |your )?assignment\b.*"
    r"|.*\bother specialists (?:will )?handle\b.*"
    r")\s*$"
)
_BARE_INDEX = re.compile(r"^\s*\d+\s*$")


def clean_specialist_text(text: str) -> str:
    """Drop slice-refusal / fake walkthrough lines; keep real answer lines."""
    if not text:
        return ""
    lines = text.splitlines()
    drop: set[int] = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _REFUSAL_LINE.match(stripped):
            drop.add(i)
            # Also drop a bare "12" index line immediately above the refusal.
            if i > 0 and _BARE_INDEX.match(lines[i - 1].strip()):
                drop.add(i - 1)
            continue
    kept = [line for i, line in enumerate(lines) if i not in drop]
    # Second pass: orphan bare numbers left between content.
    cleaned_lines: list[str] = []
    for i, line in enumerate(kept):
        if _BARE_INDEX.match(line.strip()):
            # Keep bare numbers only if they look like list content inside an answer
            # (followed by non-refusal prose on next line that isn't empty).
            nxt = kept[i + 1].strip() if i + 1 < len(kept) else ""
            if not nxt or _REFUSAL_LINE.match(nxt) or _BARE_INDEX.match(nxt):
                continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
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
