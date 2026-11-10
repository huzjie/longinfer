# -*- coding: utf-8 -*-
"""视觉分词器：patch -> 特征投影 -> 多级 RVQ 离散 token。"""
from __future__ import annotations

from typing import List, Sequence

from .patchify import patchify
from .rvq import ResidualVQ
from ..utils.tensor import mat_vec, randn


class VisionTokenizer:
    """把图像转成多层级离散 token（与文本 token 同空间、共享主干）。"""

    def __init__(self, patch_size: int = 16, code_dim: int = 64, rvq_levels: int = 4,
                 codebook_size: int = 512, seed: int = 0):
        self.patch_size = patch_size
        self.code_dim = code_dim
        self.rvq_levels = rvq_levels
        # 投影矩阵：patch 向量 -> code_dim
        self.proj = randn((code_dim, patch_size * patch_size), seed=seed + 42)
        self.rvq = ResidualVQ(rvq_levels, codebook_size, code_dim, seed=seed)

    def encode(self, image: Sequence[Sequence[float]]):
        """返回 (tokens 二维列表, 每 patch 的多级码下标)。"""
        patches = patchify(image, self.patch_size)
        tokens = []
        for p in patches:
            feat = mat_vec(self.proj, p)
            codes = self.rvq.encode(feat)
            tokens.append(codes)
        return tokens

    def num_patches(self, image: Sequence[Sequence[float]]) -> int:
        h = len(image)
        w = len(image[0]) if h else 0
        return (h // self.patch_size) * (w // self.patch_size)


def image_to_tokens(image: Sequence[Sequence[float]], patch_size: int = 16, rvq_levels: int = 4) -> List[List[int]]:
    """便捷函数：图像 -> 多级离散 token。"""
    tok = VisionTokenizer(patch_size=patch_size, rvq_levels=rvq_levels)
    return tok.encode(image)
