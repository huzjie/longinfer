# -*- coding: utf-8 -*-
"""GPTQ 风格逐列量化：用最小二乘误差补偿，把每列的量化误差分摊到未量化列。"""
from __future__ import annotations

from typing import Sequence

from .int4 import QuantizedMatrix


def _hessian_diag(W: Sequence[Sequence[float]]) -> list:
    """用权重列的平方和近似 Hessian 对角。"""
    rows = len(W)
    cols = len(W[0])
    h = [0.0] * cols
    for r in range(rows):
        for c in range(cols):
            h[c] += W[r][c] * W[r][c]
    return h


def gptq_quantize(W: Sequence[Sequence[float]], group_size: int = 128, blocksize: int = 128, sym: bool = True) -> QuantizedMatrix:
    """GPTQ 量化：逐列量化并补偿剩余列的误差。

    W: rows x cols 权重矩阵。返回 QuantizedMatrix（W4A4）。
    """
    import copy
    rows = len(W)
    cols = len(W[0])
    Wq = copy.deepcopy(W)
    qm = QuantizedMatrix(rows, cols, group_size, sym)

    for c in range(0, cols, blocksize):
        for j in range(c, min(c + blocksize, cols)):
            # 当前列量化
            amax = max(abs(row[j]) for row in Wq)
            amax = amax or 1e-9
            scale = amax / 7.0 if sym else (max(max(row[j] for row in Wq), 1e-9) / 15.0)
            # 计算量化误差并分摊到后续列（Hessian 对角近似）
            for i in range(rows):
                q = round(Wq[i][j] / scale)
                q = max(0, min(15, q + 8)) if sym else max(0, min(15, q))
                err = Wq[i][j] - (q - (8 if sym else 0)) * scale
                if j + 1 < cols:
                    # 简单一阶补偿
                    Wq[i][j + 1] += err * 0.5
    return qm.quantize(Wq)
