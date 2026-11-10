# -*- coding: utf-8 -*-
"""分片计划：把权重/专家/词表按并行维度切分。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ShardingPlan:
    world_size: int = 1
    tensor_parallel: int = 1
    expert_parallel: int = 1
    vocab_parallel: int = 1
    expert_ids_per_rank: List[List[int]] = field(default_factory=list)
    vocab_shards_per_rank: List[tuple] = field(default_factory=list)

    def summary(self) -> str:
        return (f"world={self.world_size} TP={self.tensor_parallel} "
                f"EP={self.expert_parallel} VP={self.vocab_parallel}")


def plan_sharding(num_experts: int, world_size: int, tensor_parallel: int = 1,
                  expert_parallel: int = 1, vocab_parallel: int = 1, vocab_size: int = 32000) -> ShardingPlan:
    """生成分片计划：专家按 EP 均分到各 rank，词表按 VP 切分。"""
    plan = ShardingPlan(world_size, tensor_parallel, expert_parallel, vocab_parallel)
    ep = max(1, expert_parallel)
    for rank in range(world_size):
        start = (num_experts // ep) * (rank % ep)
        end = start + (num_experts // ep) if rank % ep < num_experts % ep else start + (num_experts // ep)
        plan.expert_ids_per_rank.append(list(range(start, min(end, num_experts))))
    vp = max(1, vocab_parallel)
    shard = (vocab_size + vp - 1) // vp
    for rank in range(world_size):
        s = (rank % vp) * shard
        plan.vocab_shards_per_rank.append((s, min(s + shard, vocab_size)))
    return plan
