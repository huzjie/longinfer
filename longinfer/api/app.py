# -*- coding: utf-8 -*-
"""FastAPI 应用工厂。"""
from __future__ import annotations

from ..config import Config
from .middleware import AuthMiddleware, ObservabilityMiddleware
from .inference import InferenceService


def create_app(config: Config | None = None):
    try:
        from fastapi import FastAPI
    except Exception as e:  # noqa
        raise RuntimeError("缺少 fastapi，请安装: pip install longinfer[api]") from e

    config = config or Config.discover()
    app = FastAPI(title="longinfer", version="0.1.0",
                  description="长程多模态 MoE 智能体推理与部署引擎")
    app.state.config = config
    app.state.service = InferenceService(config)
    app.add_middleware(ObservabilityMiddleware)
    app.add_middleware(AuthMiddleware)

    from .routers import health, chat, vision, agent, quant, cluster, web
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(vision.router)
    app.include_router(agent.router)
    app.include_router(quant.router)
    app.include_router(cluster.router)
    app.include_router(web.router)

    @app.get("/")
    def root():
        return {"service": "longinfer", "docs": "/docs", "health": "/health"}

    return app
