"""Rule-based prompt decomposition into specialist-ready parts.

Never invents text that is not in the original prompt — only splits and tags.
"""
from __future__ import annotations

import re
from typing import Optional

from app.schemas import PromptPart, TaskType

_HEADING = re.compile(
    r"(?m)^(?:#{1,3}\s+.+|[A-Z][A-Z0-9 _/-]{2,40}:\s*$|"
    r"(?:\d+[\).]\s*|[A-E][\).]\s*|[-*•]\s+)\S.+)$"
)
# Optional space after )/. so "4)A tank" and "12): Describe" still split.
# Also "11Friendly" (missing paren) via digits immediately followed by a letter.
_NUMBERED_SPLIT = re.compile(
    r"(?m)(?=^(?:"
    r"\d+[\).]\s*"
    r"|\d+(?=[A-Za-z])"
    r"|[A-E][\).]\s*"
    r"|#{1,3}\s+"
    r"|(?:REASONING|CODE|QA|SUMMARY|HAIKU|MATH|VOCAB|DEFINE|DEBUG|"
    r"TRANSLAT|EXTRACT|ANALYSIS|VISION|CHAT)\b"
    r"))"
)
_LEADING_LABEL = re.compile(
    r"(?is)^\s*(?:\d+[\).]\s*:?\s*|\d+(?=[A-Za-z]))?"
    r"(REASONING|TRANSLATION|TRANSLAT|CODE|DEBUG|MATH|QA|SUMMARY|SUMMARIZATION|"
    r"CREATIVE|EXTRACTION|EXTRACT|ANALYSIS|VISION|CHAT)\s*[:\-–]\s*"
)
_LABEL_TO_TASK: dict[str, TaskType] = {
    "REASONING": "reasoning",
    "TRANSLATION": "translation",
    "TRANSLAT": "translation",
    "CODE": "code",
    "DEBUG": "debug",
    "MATH": "math",
    "QA": "qa",
    "SUMMARY": "summarization",
    "SUMMARIZATION": "summarization",
    "CREATIVE": "creative",
    "EXTRACTION": "extraction",
    "EXTRACT": "extraction",
    "ANALYSIS": "analysis",
    "VISION": "vision",
    "CHAT": "chat",
}

# More specific types first when classifying.
_VISION = re.compile(
    r"\b(image|screenshot|photo|picture|diagram|describe (?:this |the )?(?:image|photo|screenshot)|"
    r"vision|what do you see|alt text|dashboard chart comparing)\b",
    re.I,
)
_TRANSLATION = re.compile(
    r"\b(translate|translation|in (?:spanish|french|german|hindi|japanese|chinese|arabic)|"
    r"to (?:spanish|french|german|hindi|japanese|chinese|arabic)|翻译|traduci|übersetz)\b",
    re.I,
)
_EXTRACTION = re.compile(
    r"\b(extract|extraction|as json|to json|json object|return json|entities|named entities|"
    r"fill (?:a |the )?table|structured data|key[- ]value)\b",
    re.I,
)
_DEBUG = re.compile(
    r"\b(fix (?:this|the|my)|debug|traceback|stack trace|typeerror|syntaxerror|"
    r"exception|bug|broken code|doesn'?t compile|failing test|"
    r"why does .+ return|map\(parseInt\))\b",
    re.I,
)
_ANALYSIS = re.compile(
    r"\b(analyze|analysis|compare|comparison|critique|pros and cons|trade-?offs?|"
    r"evaluate|benchmark|versus| vs\.? |rank these)\b",
    re.I,
)
_MATH = re.compile(
    r"\b(calculate|compute|equation|integral|derivative|algebra|geometry|"
    r"probability|statistics|prove that|solve for|math|mathematical|"
    r"which number is larger|which is larger|numeric|fraction|percentage|"
    r"find the capacity|liters?|litres?|tank is)\b|"
    r"\d+\s*[\+\-\*/÷×]\s*\d+|=\s*\?|solve\s+\d",
    re.I,
)
_CODE = re.compile(
    r"\b(python|typescript|javascript|sql|regex|function|def |class |compile|"
    r"fastapi|endpoint|linked list|binary search|implement|"
    r"doctest|write a (?:python )?function|code|"
    r"request_logs|select .+ from|parseInt)\b",
    re.I,
)
_SUM = re.compile(
    r"\b(summarize|summary|tldr|tl;dr|short summary|executive brief|compress)\b",
    re.I,
)
_REASON = re.compile(
    r"\b(step by step|prove|explain why|logic|argue|verdict|for and against|"
    r"think step|reasoning|condition|who owns the fish|logic sketch)\b",
    re.I,
)
_CREATIVE = re.compile(
    r"\b(poem|haiku|story|fictional|fable|invent|sci-fi|free-?verse|"
    r"product myth|write a (?:four|4|8)-line)\b",
    re.I,
)
_CHAT = re.compile(
    r"\b(hi+|hello|hey there|what'?s up|how are you|good morning|good evening|"
    r"chit-?chat|casual (?:chat|reply)|just checking in|"
    r"welcome|friendly \d+-sentence|viva evaluator)\b",
    re.I,
)
_VOCAB_QA = re.compile(
    r"\b(define|definition|vocabulary|what (?:is|are|does)|contrast|difference|"
    r"meaning of|in one (?:short )?paragraph)\b",
    re.I,
)


def classify_segment(text: str, hint: Optional[TaskType] = None) -> tuple[TaskType, str]:
    if hint:
        return hint, "client_hint"
    # Explicit leading labels win (e.g. "16) REASONING: … translation …").
    label_m = _LEADING_LABEL.match(text.strip())
    if label_m:
        key = label_m.group(1).upper()
        task = _LABEL_TO_TASK.get(key)
        if task:
            return task, "leading_label"
    # Specific → general
    if _VISION.search(text):
        return "vision", "vision_keywords"
    if _TRANSLATION.search(text):
        return "translation", "translation_keywords"
    if _EXTRACTION.search(text):
        return "extraction", "extraction_keywords"
    if _DEBUG.search(text):
        return "debug", "debug_keywords"
    if _ANALYSIS.search(text):
        return "analysis", "analysis_keywords"
    if _CODE.search(text):
        return "code", "code_keywords"
    if _MATH.search(text):
        return "math", "math_keywords"
    if _SUM.search(text):
        return "summarization", "summarization_keywords"
    if _REASON.search(text):
        return "reasoning", "reasoning_keywords"
    if _CREATIVE.search(text):
        return "creative", "creative_keywords"
    if _CHAT.search(text):
        return "chat", "chat_keywords"
    if _VOCAB_QA.search(text):
        return "qa", "qa_keywords"
    return "qa", "default_qa"


# Split blended one-paragraph asks on clause boundaries that introduce a new intent.
_MULTI_INTENT = re.compile(
    r"(?:"
    r",\s*(?=then\b)|"
    r",\s*(?=and\s+(?:then\b|also\b|finally\b|finish\b|end\b|contrast\b|"
    r"compress\b|summarize\b|tell\b|write\b|define\b|calculate\b|explain\b|"
    r"translate\b|extract\b|analyze\b|debug\b|fix\b|compare\b))|"
    r",\s*(?=contrast\b)|"
    r",\s*(?=compress\b)|"
    r",\s*(?=summarize\b)|"
    r",\s*(?=translate\b)|"
    r",\s*(?=extract\b)|"
    r",\s*(?=analyze\b)|"
    r",\s*(?=compare\b)|"
    r",\s*(?=debug\b)|"
    r",\s*(?=fix\b)|"
    r",\s*(?=finish\s+with\b)|"
    r",\s*(?=end\s+with\b)|"
    r",\s*(?=tell me\b)|"
    r",\s*(?=which (?:number )?is larger\b)|"
    r"\s+and\s+(?=finish\s+with\b)|"
    r"\s+and\s+(?=end\s+with\b)|"
    r"\s+and\s+(?=then\b)|"
    r"\s+then\s+(?=tell\b|write\b|explain\b|define\b|contrast\b|calculate\b|"
    r"compress\b|summarize\b|which\b|translate\b|extract\b|analyze\b|fix\b)|"
    r"(?<=[.!?])\s+(?=Also\b|Then\b|Next\b|Finally\b|Additionally\b|"
    r"Write\b|Calculate\b|Define\b|Summarize\b|Prove\b|Compose\b|Contrast\b|"
    r"Compress\b|Tell\b|Translate\b|Extract\b|Analyze\b|Fix\b|Debug\b|Compare\b)"
    r")",
    re.I,
)


def _split_multi_intent(text: str) -> list[str]:
    """Clause-level split for single-paragraph multi-ask prompts."""
    chunks = [c.strip(" ,") for c in _MULTI_INTENT.split(text) if c and c.strip(" ,")]
    if len(chunks) < 2:
        return []
    types = {classify_segment(c)[0] for c in chunks}
    if len(types) >= 2 or len(chunks) >= 3:
        return chunks
    return []


def _split_blocks(prompt: str) -> list[str]:
    text = prompt.strip()
    if not text:
        return []

    pieces = [p.strip() for p in _NUMBERED_SPLIT.split(text) if p and p.strip()]
    if len(pieces) >= 2:
        return pieces

    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paras) >= 2:
        merged: list[str] = []
        i = 0
        while i < len(paras):
            cur = paras[i]
            if (
                i + 1 < len(paras)
                and len(cur) < 120
                and not _HEADING.match(cur)
                and classify_segment(cur)[0] in {"qa", "chat"}
                and classify_segment(paras[i + 1])[0] not in {"qa", "chat"}
            ):
                merged.append(cur + "\n\n" + paras[i + 1])
                i += 2
            else:
                merged.append(cur)
                i += 1
        if len(merged) >= 2:
            return merged

    cue_split = re.split(
        r"(?<=[.!?])\s+(?=(?:Also|Then|Next|Additionally|Finally|"
        r"Write |Calculate |Define |Summarize |Prove |Compose |Contrast |"
        r"Compress |Tell |Translate |Extract |Analyze |Fix |Debug |Compare ))",
        text,
        flags=re.I,
    )
    cue_split = [c.strip() for c in cue_split if c.strip()]
    if len(cue_split) >= 2:
        return cue_split

    multi = _split_multi_intent(text)
    if multi:
        return multi

    return [text]


_INSTRUCTION_ONLY = re.compile(
    r"(?is)^\s*(?:do every numbered section.*(?:keep the headers)?|"
    r"keep the headers\.?|"
    r"answer each (?:section|part) separately\.?|"
    r"do all of the following(?: as clearly labeled sections)?\.?)\s*$"
)
_LEADING_INDEX = re.compile(r"^\s*\d+[\).]\s*:?\s*")


def _is_instruction_only(text: str) -> bool:
    return bool(_INSTRUCTION_ONLY.match(text.strip()))


def _strip_leading_index(text: str) -> str:
    return _LEADING_INDEX.sub("", text.strip(), count=1).strip()


def decompose(prompt: str, hint: Optional[TaskType] = None) -> list[PromptPart]:
    """Split prompt into tagged parts. Single-part prompts return one entry."""
    blocks = _split_blocks(prompt)
    # Drop "Do EVERY numbered section..." style preambles when other blocks exist.
    if len(blocks) >= 2:
        blocks = [b for b in blocks if not _is_instruction_only(b)]
        if not blocks:
            blocks = _split_blocks(prompt)
    parts: list[PromptPart] = []
    for i, block in enumerate(blocks):
        part_hint = hint if len(blocks) == 1 else None
        task, rationale = classify_segment(block, part_hint)
        parts.append(
            PromptPart(
                id=f"p{i + 1}",
                text=block,
                task_type=task,
                rationale=rationale,
            )
        )
    return [p for p in parts if p.text.strip()]


def specialist_prompt(
    part: PromptPart,
    *,
    original: str = "",
    sibling_summaries: list[str] | None = None,
) -> str:
    """Build the prompt sent to a specialist model for one part.

    Multi-part orchestra (non-empty sibling_summaries): isolate — send only the
    assigned slice. Including the full original request causes models to walk
    other numbered sections and emit refusal walkthrough junk.
    """
    siblings = sibling_summaries or []
    task = part.task_type
    body = _strip_leading_index((part.text or "").strip())

    if siblings:
        return (
            f"You are a specialist for task_type={task}.\n"
            f"Your entire reply must answer ONLY the single TASK below.\n"
            f"Rules:\n"
            f"- Output the answer itself — nothing else.\n"
            f"- Do not invent a numbered list of other tasks.\n"
            f"- Do not write \"this section is for…\", \"handled separately\", "
            f"\"not part of this slice\", or similar meta lines.\n"
            f"- Do not translate, extract, or solve anything outside the TASK.\n"
            f"- Be complete and concise; finish your answer.\n\n"
            f"TASK:\n{body}"
        )

    # Single-part: the slice is the full user ask.
    return (
        f"Answer the following completely and concisely "
        f"(task_type={task}). Finish your answer; do not stop mid-sentence.\n\n"
        f"{body or original.strip()}"
    )
