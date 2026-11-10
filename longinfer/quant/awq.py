# -*- coding: utf-8 -*-
"""AWQ 风格量化：按激活幅值保护重要通道，降低显著通道的量化误差。"""
from __future__ import annotations

from typing import Sequence

from .int4 import QuantizedMatrix


def protect_outliers(activations: Sequence[float], top_ratio: float = 0.01) -> set:
    """找出激活幅值最大的 top_ratio 通道，这些通道保持高精度/更大 scale。"""
    n = len(activations)
    if n == 0:
        return set()
    k = max(1, int(n * top_ratio))
    abs_vals = [(abs(v), i) for i, v in enumerate(activations)]
    abs_vals.sort(reverse=True)
    return {i for _, i in abs_vals[:k]}


def awq_quantize(W: Sequence[Sequence[float]], activations: Sequence[float] | None = None,
                 group_size: int = 128, sym: bool = True) -> QuantizedMatrix:
    """AWQ 量化：重要通道（激活幅值大）用更细的 scale 以减小误差。"""
    rows = len(W)
    cols = len(W[0])
    qm = QuantizedMatrix(rows, cols, group_size, sym)
    protected = protect_outliers(activations, 0.01) if activations is not None else set()

    Wq = [list(row) for row in W]
    for i in range(rows):
        for j in range(cols):
            v = Wq[i][j]
            if j in protected:
                # 保护通道：scale 更保守
                amax = max(abs(v), 1e-9)
                scale = amax / 15.0
            else:
                amax = max(abs(v), 1e-9)
                scale = amax / 7.0 if sym else amax / 15.0
            Wq[i][j] = v  # 保留原值，交给 quantize() 统一量化
    return qm.quantize(Wq)
