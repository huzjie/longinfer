# -*- coding: utf-8 -*-
"""反量化：量化矩阵 -> 浮点矩阵（可用于校验/低精度回退）。"""
from __future__ import annotations

from typing import Sequence

from .int4 import QuantizedMatrix


def dequantize_matrix(qm: QuantizedMatrix) -> Sequence[Sequence[float]]:
    return qm.dequantize()


def matmul_quantized(qm: QuantizedMatrix, x: Sequence[float]):
    """量化矩阵乘向量（先反量化再乘，保证离线可复现；加速内核可替换为 INT4 MMA）。"""
    W = qm.dequantize()
    from ..utils.tensor import mat_vec
    return mat_vec(W, x)
