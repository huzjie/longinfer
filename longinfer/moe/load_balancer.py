# -*- coding: utf-8 -*-
"""负载均衡：辅助损失 + 容量约束，防止专家「偏科」或过载。"""
from __future__ import annotations

import math
from typing import Sequence


def aux_load_balance_loss(routes: Sequence[Sequence[int]], num_experts: int) -> float:
    """标准 switch-transformer 辅助损失：鼓励 token 均匀分配到专家。

    loss = num_experts * sum(f_i * P_i)，其中 f_i 是专家 i 的 token 占比，P_i 是路由概率均值。
    """
    n = len(routes)
    if n == 0:
        return 0.0
    f = [0.0] * num_experts
    p = [0.0] * num_experts
    for route in routes:
        for e in route:
            f[e] += 1.0
    for i in range(num_experts):
        f[i] /= n
        p[i] = f[i]  # 简化：无显式概率输入时用频率近似
    return float(num_experts) * sum(f[i] * p[i] for i in range(num_experts))


class LoadBalancer:
    """容量约束：限制每个专家单步最多处理 capacity 个 token，超出的丢到残差。"""

    def __init__(self, num_experts: int, capacity_factor: float = 1.25):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

    def capacity(self, tokens_per_batch: int, top_k: int) -> int:
        """每个专家的容量上限。"""
        base = tokens_per_batch / self.num_experts
        return max(1, int(math.ceil(base * self.capacity_factor * top_k)))

    def enforce(self, routes: Sequence[Sequence[int]], capacity: int):
        """对超容量专家做截断，返回 (过滤后路由, 溢出 token 数)。"""
        counts = [0] * self.num_experts
        overflow = 0
        out = []
        for route in routes:
            keep = []
            for e in route:
                if counts[e] < capacity:
                    keep.append(e)
                    counts[e] += 1
                else:
                    overflow += 1
            out.append(keep)
        return out, overflow
