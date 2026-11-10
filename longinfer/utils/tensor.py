# -*- coding: utf-8 -*-
"""纯 Python（numpy-free）的矩阵/张量辅助：MoE 路由、量化、RVQ 都建立在其上。

数据表示统一为 list[list[float]]（2D 矩阵）或 list[float]（向量）。
规模较大时建议接入 numpy/加速内核，此处保证离线可复现。
"""
from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence


def zeros(rows: int, cols: int) -> List[List[float]]:
    return [[0.0] * cols for _ in range(rows)]


def mat_vec(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> List[float]:
    n = len(mat)
    m = len(vec)
    out = [0.0] * n
    for i in range(n):
        row = mat[i]
        s = 0.0
        for j in range(m):
            s += row[j] * vec[j]
        out[i] = s
    return out


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    t = max(temperature, 1e-9)
    scaled = [x / t for x in logits]
    mx = max(scaled)
    exps = [math.exp(x - mx) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def topk_indices(values: Sequence[float], k: int) -> List[int]:
    k = min(k, len(values))
    indexed = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return indexed[:k]


def norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def normalize(vec: Sequence[float]) -> List[float]:
    n = norm(vec)
    if n == 0:
        return list(vec)
    return [x / n for x in vec]


def randn(shape: Iterable[int], seed: int | None = None) -> List:
    rng = random.Random(seed)
    def _build(dims):
        if not dims:
            return rng.gauss(0.0, 1.0)
        return [_build(dims[1:]) for _ in range(dims[0])]
    return _build(tuple(shape))


def hadamard(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x * y for x, y in zip(a, b)]
