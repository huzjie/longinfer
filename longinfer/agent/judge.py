# -*- coding: utf-8 -*-
"""轨迹评判器：用 LLM 对重复推理、冗余工具调用、过度思辨施加约束。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Verdict:
    score: float          # 0~1
    excessive_thinking: bool
    redundant_tool_call: bool
    repeated_reasoning: bool
    comment: str = ""


class TrajectoryJudge:
    """LLM 评判员：包装一个 provider，产出结构化评判；无 provider 时用启发式兜底。"""

    def __init__(self, provider=None):
        self.provider = provider

    def evaluate(self, thought: str, action: str, history_len: int) -> Verdict:
        excessive = len(thought) > 400 and "我认为" in thought
        redundant = action == "" and history_len > 3
        repeated = thought in self._recent and self._recent.count(thought) > 1 if hasattr(self, "_recent") else False
        if self.provider is not None:
            try:
                prompt = (
                    "你是轨迹评判员。对下面这一步给出评分(0~1)并标记问题：\n"
                    f"思考:{thought}\n动作:{action}\n"
                    "输出 JSON: {\"score\": 0.8, \"excessive_thinking\": false, \"redundant_tool_call\": false, \"repeated_reasoning\": false, \"comment\": \"...\"}"
                )
                raw = self.provider.complete(prompt)
                import json
                data = json.loads(raw)
                return Verdict(**data)
            except Exception:
                pass
        score = 1.0 - (0.2 if excessive else 0) - (0.15 if redundant else 0) - (0.1 if repeated else 0)
        return Verdict(score=max(score, 0.0), excessive_thinking=excessive,
                       redundant_tool_call=redundant, repeated_reasoning=repeated)
