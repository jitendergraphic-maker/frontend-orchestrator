from typing import Literal, Optional

from pydantic import BaseModel, Field

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


class CompleteRequest(BaseModel):
    prompt: str = Field(min_length=1)
    task_type: TaskType | None = None
    policy: Policy = "orchestra"
    force_model: str | None = None


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
