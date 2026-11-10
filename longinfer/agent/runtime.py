# -*- coding: utf-8 -*-
"""智能体运行时：感知-思考-行动-观测闭环 + 检查点 + 评判器。"""
from __future__ import annotations

import time
from typing import Optional

from .trajectory import Trajectory, Step
from .memory import AgentMemory
from .judge import TrajectoryJudge
from .tools import ToolRegistry
from .rollout import PartialRollout
from .checkpoint import save_checkpoint, load_checkpoint
from .state import AgentState
from ..config import Config
from ..logging import get_logger

log = get_logger("longinfer.agent")


class AgentRuntime:
    def __init__(self, config: Config, provider=None, tools: Optional[ToolRegistry] = None):
        self.config = config
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.memory = AgentMemory()
        acfg = config["agent"]
        self.max_steps = acfg.get("max_steps", 200)
        self.checkpoint_every = acfg.get("checkpoint_every", 10)
        self.rollout = PartialRollout(self.checkpoint_every, acfg.get("oversample_by_pass_rate", True))
        self.judge = TrajectoryJudge(provider if acfg.get("judge_provider") else None)
        self.state = AgentState.IDLE
        self.checkpoint_dir = config.resolve("checkpoint_dir", "~/.longinfer/checkpoints")

    def _think(self, goal: str, obs: str) -> tuple[str, str]:
        if self.provider is None:
            return f"针对目标「{goal}」结合观察「{obs}」进行推理", "reason"
        prompt = f"目标: {goal}\n当前观察: {obs}\n请输出下一步：先思考，再给出动作。"
        return self.provider.complete(prompt), "act"

    def run(self, goal: str, resume_from: Optional[str] = None) -> Trajectory:
        traj = load_checkpoint(resume_from) if resume_from else Trajectory(goal=goal)
        start = len(traj.steps)
        self.state = AgentState.RUNNING
        obs = ""
        for i in range(start, self.max_steps):
            thought, action = self._think(goal, obs)
            verdict = self.judge.evaluate(thought, action, len(traj.steps))
            tool, tool_input = self._resolve_tool(action)
            result = self._execute(tool, tool_input)
            obs = result
            self.memory.add_observation(result)
            step = Step(step_id=i, thought=thought, action=action, tool=tool or "",
                        tool_input=tool_input, observation=result, reward=verdict.score,
                        done=self._is_done(result))
            traj.add(step)
            if self.rollout.should_checkpoint(i + 1):
                save_checkpoint(traj, f"{self.checkpoint_dir}/rollout-{int(time.time())}.json")
            if step.done:
                break
        self.state = AgentState.DONE
        return traj

    def _resolve_tool(self, action: str):
        if not action:
            return None, None
        for name in self.tools.names():
            if name in action:
                return name, {}
        return None, None

    def _execute(self, tool, tool_input):
        if tool is None:
            return "（无工具调用，纯推理步）"
        try:
            return str(self.tools.get(tool).run(**(tool_input or {})))
        except Exception as e:  # noqa
            return f"工具执行失败: {e}"

    @staticmethod
    def _is_done(result: str) -> bool:
        return any(k in (result or "") for k in ("DONE", "完成", "成功", "SUCCESS"))


def run_agent(config: Config, provider=None, tools: Optional[ToolRegistry] = None,
              goal: str = "", resume_from: Optional[str] = None) -> Trajectory:
    rt = AgentRuntime(config, provider, tools)
    return rt.run(goal, resume_from)
