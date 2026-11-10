# -*- coding: utf-8 -*-
"""专家实现：标准 FFN 专家（带 GELU 激活）。"""
from __future__ import annotations

from typing import Sequence

from ..utils.tensor import mat_vec, zeros


class Expert:
    """专家基类。"""

    def __init__(self, expert_id: int):
        self.expert_id = expert_id

    def forward(self, x: Sequence[float]) -> Sequence[float]:
        raise NotImplementedError


class FFNExpert(Expert):
    """两阶段 FFN：h = W1 @ x -> gelu -> W2 @ h。

    为保持离线可复现，权重用确定性伪随机初始化；真实部署从 checkpoint 加载 W4A4 量化权重。
    """

    def __init__(self, expert_id: int, dim: int, hidden: int, seed: int = 0):
        super().__init__(expert_id)
        self.dim = dim
        self.hidden = hidden
        from ..utils.tensor import randn
        # 每专家独立种子，避免专家权重完全相同
        self.w1 = randn((hidden, dim), seed=seed + expert_id * 1000)
        self.w2 = randn((dim, hidden), seed=seed + expert_id * 2000 + 1)

    @staticmethod
    def _gelu(x: float) -> float:
        # GELU 近似
        return 0.5 * x * (1.0 + __import__("math").tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

    def forward(self, x: Sequence[float]) -> Sequence[float]:
        h = mat_vec(self.w1, x)
        h = [self._gelu(v) for v in h]
        return mat_vec(self.w2, h)
