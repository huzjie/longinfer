# -*- coding: utf-8 -*-
"""长程智能体编排。"""
from __future__ import annotations

from fastapi import APIRouter, Request
from . import schemas

router = APIRouter(tags=["agent"])


@router.post("/v1/agent/run", response_model=schemas.AgentResponse)
def run(req: schemas.AgentRequest, request: Request):
    from ...agent.runtime import AgentRuntime
    from ...tools import build_default_registry
    svc = request.app.state.service
    rt = AgentRuntime(request.app.state.config, svc.provider, build_default_registry())
    traj = rt.run(req.goal, req.max_steps)
    return {"goal": traj.goal, "steps": len(traj.steps), "trajectory": [s.to_dict() for s in traj.steps[-20:]]}
