# -*- coding: utf-8 -*-
"""专家并行与动态专家复制（MoonEP 式）：热专家多副本，冷专家合并。"""
from __future__ import annotations

from typing import List, Sequence


class ExpertParallel:
    def __init__(self, num_experts: int, num_ranks: int):
        self.num_experts = num_experts
        self.num_ranks = num_ranks
        self.replicas: dict[int, int] = {e: 1 for e in range(num_experts)}

    def set_replicas(self, replicas: dict[int, int]) -> None:
        self.replicas.update(replicas)

    def placement(self) -> dict[int, List[int]]:
        """返回 专家 -> 所在 rank 列表（按副本数）。"""
        place: dict[int, List[int]] = {}
        rank_cursor = 0
        for e in sorted(self.replicas, key=lambda x: -self.replicas[x]):
            reps = self.replicas[e]
            place[e] = []
            for _ in range(reps):
                place[e].append(rank_cursor % self.num_ranks)
                rank_cursor += 1
        return place


def dynamic_replication(hot_counts: Sequence[int], num_ranks: int, total_capacity: int) -> dict[int, int]:
    """按专家访问热度动态复制：热门专家分配更多副本，冷门专家合并到单副本。

    hot_counts: 每个专家近期的 token 访问计数。
    total_capacity: 可用副本总数（预算），各专家至少 1 副本，剩余按热度最大余数法分配。
    """
    n = len(hot_counts)
    if n == 0:
        return {}
    total = sum(hot_counts) or 1
    replicas = {e: 1 for e in range(n)}
    remaining = total_capacity - n
    if remaining <= 0:
        return replicas  # 容量不足，只能各 1 个
    weights = [hot_counts[e] / total for e in range(n)]
    raw = [w * remaining for w in weights]
    alloc = [int(r) for r in raw]
    rem = remaining - sum(alloc)
    # 余数大的优先补 1
    frac = sorted(((raw[e] - alloc[e], e) for e in range(n)), reverse=True)
    for i in range(rem):
        alloc[frac[i][1]] += 1
    for e in range(n):
        replicas[e] = 1 + alloc[e]
    return replicas
