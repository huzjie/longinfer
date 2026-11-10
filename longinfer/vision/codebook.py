# -*- coding: utf-8 -*-
"""视觉码本：与主干联合训练的离散码向量集合。"""
from __future__ import annotations

from typing import List, Sequence

from ..utils.tensor import randn, dot, normalize


class Codebook:
    """离散码本：codebook_size 个 code_dim 维码向量。"""

    def __init__(self, codebook_size: int, code_dim: int, seed: int = 0):
        self.size = codebook_size
        self.dim = code_dim
        self.codes: List[List[float]] = [normalize(v) for v in randn((codebook_size, code_dim), seed=seed)]

    def nearest(self, vec: Sequence[float]) -> int:
        """最近邻检索：返回与 vec 最接近的码向量下标。"""
        best = -1
        best_score = -1e18
        for i, code in enumerate(self.codes):
            s = dot(vec, code)
            if s > best_score:
                best_score = s
                best = i
        return best

    def decode(self, idx: int) -> List[float]:
        return list(self.codes[idx])


def init_codebook(codebook_size: int = 16384, code_dim: int = 256, seed: int = 0) -> Codebook:
    return Codebook(codebook_size, code_dim, seed=seed)
