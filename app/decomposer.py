"""Rule-based prompt decomposition into specialist-ready parts.

Never invents text that is not in the original prompt — only splits and tags.
"""
from __future__ import annotations

import re
from typing import Optional

from app.schemas import PromptPart, TaskType

_HEADING = re.compile(
    r"(?m)^(?:#{1,3}\s+.+|[A-Z][A-Z0-9 _/-]{2,40}:\s*$|"
    r"(?:\d+[\).]|[A-E][\).]|[-*•])\s+\S.+)$"
)
_NUMBERED_SPLIT = re.compile(
    r"(?m)(?=^(?:\d+[\).]\s+|[A-E][\).]\s+|#{1,3}\s+|"
    r"(?:REASONING|CODE|QA|SUMMARY|HAIKU|MATH|VOCAB|DEFINE|DEBUG|"
    r"TRANSLAT|EXTRACT|ANALYSIS|VISION|CHAT)\b))"
)

# More specific types first when classifying.
_VISION = re.compile(
    r"\b(image|screenshot|photo|picture|diagram|describe (?:this |the )?(?:image|photo|screenshot)|"
    r"vision|what do you see|alt text)\b",
    re.I,
)
_TRANSLATION = re.compile(
    r"\b(translate|translation|in (?:spanish|french|german|hindi|japanese|chinese|arabic)|"
    r"to (?:spanish|french|german|hindi|japanese|chinese|arabic)|翻译|traduci|übersetz)\b",
    re.I,
)
_EXTRACTION = re.compile(
    r"\b(extract|extraction|as json|to json|json object|entities|named entities|"
    r"fill (?:a |the )?table|structured data|key[- ]value)\b",
    re.I,
)
_DEBUG = re.compile(
    r"\b(fix (?:this|the|my)|debug|traceback|stack trace|typeerror|syntaxerror|"
    r"exception|bug|broken code|doesn'?t compile|failing test)\b",
    re.I,
)
_ANALYSIS = re.compile(
    r"\b(analyze|analysis|compare|comparison|critique|pros and cons|trade-?offs?|"
    r"evaluate|benchmark|versus| vs\.? )\b",
    re.I,
)
_MATH = re.compile(
    r"\b(calculate|compute|equation|integral|derivative|algebra|geometry|"
    r"probability|statistics|prove that|solve for|math|mathematical|"
    r"which number is larger|which is larger|numeric|fraction|percentage)\b|"
    r"\d+\s*[\+\-\*/÷×]\s*\d+|=\s*\?",
    re.I,
)
_CODE = re.compile(
    r"\b(python|typescript|javascript|sql|regex|function|def |class |compile|"
    r"fastapi|endpoint|linked list|binary search|implement|"
    r"doctest|write a (?:python )?function|code)\b",
    re.I,
)
_SUM = re.compile(
    r"\b(summarize|summary|tldr|tl;dr|short summary|executive brief|compress)\b",
    re.I,
)
_REASON = re.compile(
    r"\b(step by step|prove|explain why|logic|argue|verdict|for and against|"
    r"think step|reasoning|condition)\b",
    re.I,
)
_CREATIVE = re.compile(
    r"\b(poem|haiku|story|fictional|fable|invent|sci-fi|write a (?:four|4)-line)\b",
    re.I,
)
_CHAT = re.compile(
    r"\b(hi+|hello|hey there|what'?s up|how are you|good morning|good evening|"
    r"chit-?chat|casual (?:chat|reply)|just checking in)\b",
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


def decompose(prompt: str, hint: Optional[TaskType] = None) -> list[PromptPart]:
    """Split prompt into tagged parts. Single-part prompts return one entry."""
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
    original: str,
    sibling_summaries: list[str],
) -> str:
    """Build the prompt sent to a specialist model for one part."""
    ctx_bits = []
    clipped = original.strip()
    if len(clipped) > 400:
        clipped = clipped[:400] + "…"
    ctx_bits.append(f"Original user request (context only):\n{clipped}")
    if sibling_summaries:
        ctx_bits.append(
            "Other parts of this request (handled separately):\n- "
            + "\n- ".join(sibling_summaries)
        )
    ctx = "\n\n".join(ctx_bits)
    return (
        f"{ctx}\n\n"
        f"---\n"
        f"Your assigned slice (task={part.task_type}). "
        f"Answer ONLY this slice completely and concisely. "
        f"Do not repeat or answer other slices. Finish your answer — "
        f"do not stop mid-sentence.\n\n{part.text}"
    )
