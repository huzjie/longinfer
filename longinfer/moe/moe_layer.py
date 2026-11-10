# -*- coding: utf-8 -*-
"""MoE 层：把路由器与专家池组合，完成一次稀疏前向。"""
from __future__ import annotations

from typing import List, Sequence

from .routing import Router
from .experts import FFNExpert


class MoELayer:
    """MoE 前向：对每个 token 走 top_k 个专家，加权求和。"""

    def __init__(self, dim: int, num_experts: int = 256, top_k: int = 8, hidden_ratio: float = 4.0, seed: int = 0):
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        hidden = int(dim * hidden_ratio)
        self.router = Router(dim, num_experts, top_k, seed=seed)
        self.experts: List[FFNExpert] = [FFNExpert(i, dim, hidden, seed=seed) for i in range(num_experts)]

    def forward(self, x: Sequence[float]) -> Sequence[float]:
        idxs, probs = self.router.route(x)
        out = [0.0] * self.dim
        for idx, p in zip(idxs, probs):
            y = self.experts[idx].forward(x)
            for j in range(self.dim):
                out[j] += p * y[j]
        return out

    def active_params(self) -> int:
        """估算一次前向激活的参数量（对应「48B 激活」的口径）。"""
        per_expert = self.experts[0].dim * self.experts[0].hidden * 2
        return self.top_k * per_expert

    def total_params(self) -> int:
        """总参数量（对应「1.6T 总参」的口径）。"""
        per_expert = self.experts[0].dim * self.experts[0].hidden * 2
        return self.num_experts * per_expert + self.router.num_experts * self.dim
