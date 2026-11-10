# -*- coding: utf-8 -*-
"""图像切 patch：把 (H, W) 灰度图切成 (H//p, W//p) 个 p×p 块。"""
from __future__ import annotations

from typing import List, Sequence


def patchify(image: Sequence[Sequence[float]], patch_size: int = 16) -> List[List[float]]:
    """返回每个 patch 展平后的向量列表（长度 patch_size*patch_size）。"""
    h = len(image)
    w = len(image[0]) if h else 0
    patches: List[List[float]] = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            vec = []
            for dy in range(patch_size):
                for dx in range(patch_size):
                    vec.append(float(image[y + dy][x + dx]))
            patches.append(vec)
    return patches


def grid_shape(h: int, w: int, patch_size: int = 16):
    return (h // patch_size, w // patch_size)
