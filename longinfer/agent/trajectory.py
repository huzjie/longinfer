# -*- coding: utf-8 -*-
"""轨迹记录：每个执行步的结构化数据，支持重放与溯源。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Step:
    step_id: int
    thought: str = ""
    action: str = ""
    tool: str = ""
    tool_input: Any = None
    observation: str = ""
    reward: float = 0.0
    done: bool = False
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trajectory:
    goal: str
    steps: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict:
        return {"goal": self.goal, "steps": [s.to_dict() for s in self.steps], "created_at": self.created_at}

    def summary(self) -> str:
        return f"goal={self.goal!r} steps={len(self.steps)}"
