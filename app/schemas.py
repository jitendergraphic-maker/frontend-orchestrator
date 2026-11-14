from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

TaskType = Literal[
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
]
Policy = Literal["orchestra", "routed", "baseline"]
AttemptOutcome = Literal["success", "error", "low_confidence"]
SynthesisMethod = Literal["stitch", "none"]
ChatRole = Literal["user", "assistant", "system"]


class ChatTurn(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)


class CompleteRequest(BaseModel):
    """Single-turn prompt and/or multi-turn messages (chat follow-ups)."""

    prompt: str = ""
    messages: list[ChatTurn] | None = None
    task_type: TaskType | None = None
    policy: Policy = "orchestra"
    force_model: str | None = None

    @model_validator(mode="after")
    def require_prompt_or_messages(self) -> "CompleteRequest":
        has_messages = bool(self.messages) and any(
            m.content.strip() for m in (self.messages or [])
        )
        has_prompt = bool((self.prompt or "").strip())
        if not has_prompt and not has_messages:
            raise ValueError("prompt or messages is required")
        if has_messages:
            if not any(m.role == "user" and m.content.strip() for m in self.messages or []):
                raise ValueError("messages must include at least one user turn")
        return self


class AttemptTrace(BaseModel):
    index: int
    model: str
    provider: str
    outcome: AttemptOutcome
    reason: str | None = None
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    est_cost: float = 0.0
    part_id: str | None = None


class ScoreBreakdown(BaseModel):
    id: str
    provider: str
    display_name: str = ""
    strength: float
    cost_norm: float
    latency_norm: float
    score: float
    eligible: bool = True
    reject_reason: str | None = None


class PromptPart(BaseModel):
    id: str
    text: str
    task_type: TaskType
    rationale: str = ""


class PartOutput(BaseModel):
    part_id: str
    task_type: TaskType
    text: str
    model: str
    provider: str
    success: bool
    tokens_in: int = 0
    tokens_out: int = 0
    est_cost: float = 0.0
    latency_ms: int = 0
    attempts: list[AttemptTrace] = Field(default_factory=list)
    ranked: list[str] = Field(default_factory=list)
    scoreboard: list[ScoreBreakdown] = Field(default_factory=list)
    error: str | None = None


class CompleteResponse(BaseModel):
    output: str
    task_type: str
    model: str
    provider: str
    policy: str
    tokens_in: int
    tokens_out: int
    est_cost: float
    latency_ms: int
    success: bool
    fallback_used: bool
    fallback_from: str | None
    # Heuristic quality proxy in [0,1]; null when the request failed / was not scored.
    quality_score: Optional[float] = None
    quality_label: str = "heuristic quality proxy"
    ranked: list[str]
    attempts: list[AttemptTrace] = Field(default_factory=list)
    scoreboard: list[ScoreBreakdown] = Field(default_factory=list)
    parts: list[PromptPart] = Field(default_factory=list)
    part_outputs: list[PartOutput] = Field(default_factory=list)
    synthesis_method: SynthesisMethod = "none"


class EvalRunRequest(BaseModel):
    limit: int | None = None
    notes: str | None = None
