# -*- coding: utf-8 -*-
"""智能体状态机。"""
from __future__ import annotations

from enum import Enum


class AgentState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
