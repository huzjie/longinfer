# -*- coding: utf-8 -*-
"""部署计划：综合拓扑、模型规模与并行配置，产出可落地的部署方案。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .topology import Topology
from .sharding import plan_sharding
from .expert_parallel import dynamic_replication


@dataclass
class DeploymentPlan:
    total_params_b: float = 1600.0     # 总参（B）
    active_params_b: float = 48.0      # 激活参（B）
    world_size: int = 8
    tensor_parallel: int = 1
    expert_parallel: int = 8
    vocab_parallel: int = 2
    per_card_gb: float = 80.0
    bytes_per_param: float = 2.0       # 2 = BF16, 0.5 = W4A4
    notes: List[str] = field(default_factory=list)

    def estimated_memory_gb(self) -> float:
        return self.total_params_b * self.bytes_per_param * 1e9 / 1e9

    def per_card_usage_gb(self) -> float:
        return self.estimated_memory_gb() / self.world_size

    def summary(self) -> str:
        lines = [
            f"总参数 {self.total_params_b:.0f}B / 激活 {self.active_params_b:.0f}B",
            f"world_size={self.world_size} TP={self.tensor_parallel} EP={self.expert_parallel} VP={self.vocab_parallel}",
            f"单卡显存占用约 {self.per_card_usage_gb():.1f} GB（{self.bytes_per_param} 字节/参数）",
        ]
        return "\n".join(lines)


def build_deployment_plan(total_params_b: float, active_params_b: float, world_size: int,
                          per_card_gb: float = 80.0, quant: str = "w4a4") -> DeploymentPlan:
    bp = 0.5 if quant in ("w4a4", "int4") else 2.0
    ep = world_size if world_size > 1 else 1
    plan = DeploymentPlan(total_params_b, active_params_b, world_size, 1, ep, 2, per_card_gb, bp)
    # 单卡显存 < 80GB 时提示分片
    if plan.per_card_usage_gb() > per_card_gb:
        plan.notes.append("单卡显存不足，建议提高 EP/VP 分片或降低量化精度")
    else:
        plan.notes.append("显存充足，可直接部署")
    return plan
