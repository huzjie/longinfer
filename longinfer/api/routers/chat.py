# -*- coding: utf-8 -*-
"""对话推理。"""
from __future__ import annotations

from fastapi import APIRouter, Request
from . import schemas

router = APIRouter(tags=["chat"])


@router.post("/v1/chat", response_model=schemas.ChatResponse)
def chat(req: schemas.ChatRequest, request: Request):
    svc = request.app.state.service
    prompt = req.prompt or "\n".join(m.get("content", "") for m in req.messages if isinstance(m, dict))
    return svc.chat(prompt, req.max_new, req.temperature)
