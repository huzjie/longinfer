# -*- coding: utf-8 -*-
"""API 请求/响应模型。"""
from __future__ import annotations

try:
    from pydantic import BaseModel, Field
except Exception:  # noqa
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Field:  # type: ignore
        def __init__(self, *a, **k):
            pass


class ChatRequest(BaseModel):
    prompt: str = ""
    messages: list = Field(default_factory=list)
    max_new: int = 256
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 1.0


class ChatResponse(BaseModel):
    text: str = ""
    provider: str = ""
    model: str = ""
    usage: dict = Field(default_factory=dict)


class VisionRequest(BaseModel):
    image_path: str = ""
    patch_size: int = 16
    rvq_levels: int = 4


class VisionResponse(BaseModel):
    num_patches: int = 0
    tokens: list = Field(default_factory=list)


class QuantRequest(BaseModel):
    method: str = "w4a4"
    group_size: int = 128
    sym: bool = True


class AgentRequest(BaseModel):
    goal: str = ""
    max_steps: int = 200
    resume_from: str = ""


class AgentResponse(BaseModel):
    goal: str = ""
    steps: int = 0
    trajectory: list = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    provider: str = ""
    model: str = ""
