# -*- coding: utf-8 -*-
"""健康检查。"""
from __future__ import annotations

from fastapi import APIRouter, Request
from . import schemas

router = APIRouter(tags=["health"])


@router.get("/health", response_model=schemas.HealthResponse)
def health(request: Request):
    cfg = request.app.state.config
    return {"status": "ok", "version": "0.1.0", "provider": cfg["provider"], "model": cfg["model"]}
