# -*- coding: utf-8 -*-
"""示例 4：长程智能体运行（mock provider）。"""
from longinfer.config import Config
from longinfer.agent.runtime import AgentRuntime
from longinfer.tools import build_default_registry
from longinfer.api.inference import build_provider

config = Config()
provider = build_provider(config)
rt = AgentRuntime(config, provider, build_default_registry())
traj = rt.run("修复 checkout 页崩溃", resume_from=None)
print("步数:", len(traj.steps))
print("最终状态:", rt.state.value)
