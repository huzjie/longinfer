# -*- coding: utf-8 -*-
"""运行长程智能体。"""
from __future__ import annotations

from ..logging import get_logger

log = get_logger("longinfer.agent")


def agent_run(config, args) -> int:
    from ..agent.runtime import AgentRuntime
    from ..tools import build_default_registry
    from ..api.inference import build_provider
    provider = build_provider(config)
    rt = AgentRuntime(config, provider, build_default_registry())
    rt.max_steps = args.max_steps
    log.info("目标：%s（最多 %d 步）", args.goal, args.max_steps)
    traj = rt.run(args.goal, resume_from=args.resume)
    log.info("完成 %d 步，最终状态=%s", len(traj.steps), rt.state.value)
    for s in traj.steps[-3:]:
        print(f"  step {s.step_id}: tool={s.tool or '-'} obs={s.observation[:60]}")
    return 0
