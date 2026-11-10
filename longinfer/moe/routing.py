# -*- coding: utf-8 -*-
"""路由器：把 token 映射到 top_k 个专家。"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from ..utils.tensor import mat_vec, softmax, topk_indices


class Router:
    """线性路由器：logits = W @ x，取 top_k 专家。

    gate 权重用伪随机初始化（离线可复现），真实部署时可从 checkpoint 加载。
    """

    def __init__(self, dim: int, num_experts: int, top_k: int = 8, jitter: float = 0.0, seed: int = 0):
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter = jitter
        # 伪随机门控权重，维度 num_experts x dim
        from ..utils.tensor import randn
        self.gate = randn((num_experts, dim), seed=seed)

    def logits(self, x: Sequence[float]) -> List[float]:
        return mat_vec(self.gate, x)

    def route(self, x: Sequence[float]) -> Tuple[List[int], List[float]]:
        """返回 (选中专家下标, 归一化后的路由权重)。"""
        logits = self.logits(x)
        if self.jitter > 0.0:
            import random
            logits = [v + random.uniform(-self.jitter, self.jitter) for v in logits]
        idxs = topk_indices(logits, self.top_k)
        selected = [logits[i] for i in idxs]
        probs = softmax(selected)
        return idxs, probs
