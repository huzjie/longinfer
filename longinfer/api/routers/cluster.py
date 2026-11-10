# -*- coding: utf-8 -*-
"""分布式部署计划。"""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["cluster"])


@router.get("/v1/cluster/plan")
def plan(request: Request):
    from ...dist.plan import build_deployment_plan
    dcfg = request.app.state.config["dist"]
    p = build_deployment_plan(1600.0, 48.0, dcfg.get("world_size", 8),
                              per_card_gb=80.0, quant=request.app.state.config["quant"]["method"])
    return {"summary": p.summary(), "notes": p.notes}
