"""A2A protocol data models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    """A2A task lifecycle states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class A2ATaskRequest:
    """Incoming A2A task request."""

    skill_name: str
    input_text: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> A2ATaskRequest:
        return cls(
            skill_name=data["skill_name"],
            input_text=data["input_text"],
            task_id=data.get("task_id", uuid.uuid4().hex[:12]),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "skill_name": self.skill_name,
            "input_text": self.input_text,
            "metadata": self.metadata,
        }


@dataclass
class A2ATaskResponse:
    """A2A task response."""

    task_id: str
    status: TaskStatus
    output: str = ""
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> A2ATaskResponse:
        return cls(
            task_id=data["task_id"],
            status=TaskStatus(data["status"]),
            output=data.get("output", ""),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "metadata": self.metadata,
        }
        if self.error:
            d["error"] = self.error
        return d
