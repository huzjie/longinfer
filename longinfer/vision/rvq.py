# -*- coding: utf-8 -*-
"""残差向量量化（RVQ）：逐级量化残差，层级越多细节越丰富。"""
from __future__ import annotations

from typing import List, Sequence

from .codebook import Codebook


class ResidualVQ:
    """多级残差量化：level 0 量化原始向量，后续层级量化上一级残差。"""

    def __init__(self, levels: int, codebook_size: int, code_dim: int, seed: int = 0):
        self.levels = levels
        self.codebooks: List[Codebook] = [
            Codebook(codebook_size, code_dim, seed=seed + lvl * 777) for lvl in range(levels)
        ]

    def encode(self, vec: Sequence[float]) -> List[int]:
        """返回每层的码下标序列。"""
        residual = list(vec)
        codes: List[int] = []
        for cb in self.codebooks:
            idx = cb.nearest(residual)
            codes.append(idx)
            decoded = cb.decode(idx)
            residual = [r - d for r, d in zip(residual, decoded)]
        return codes

    def decode(self, codes: Sequence[int]) -> List[float]:
        """用码下标重建向量（多级残差累加）。"""
        out = [0.0] * self.codebooks[0].dim
        for idx, cb in zip(codes, self.codebooks):
            decoded = cb.decode(idx)
            out = [o + d for o, d in zip(out, decoded)]
        return out
