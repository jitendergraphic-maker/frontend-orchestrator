from __future__ import annotations

from typing import Optional

from app.decomposer import classify_segment
from app.schemas import TaskType


def classify(prompt: str, hint: Optional[TaskType] = None) -> TaskType:
    """Single-label classify for routed/baseline; orchestra uses decompose()."""
    task, _ = classify_segment(prompt, hint)
    return task
