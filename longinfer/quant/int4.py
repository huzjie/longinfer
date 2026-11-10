# -*- coding: utf-8 -*-
"""INT4 位打包/解包：每字节存两个 4bit 值。

约定：低 4bit 存偶下标元素，高 4bit 存奇下标元素。
"""
from __future__ import annotations

from typing import List, Sequence


def _clamp(v: int) -> int:
    return max(0, min(15, v))


def pack_int4(values: Sequence[int]) -> bytearray:
    """把 [0,15] 的整数序列打包为 4bit 字节流。"""
    out = bytearray()
    n = len(values)
    for i in range(0, n, 2):
        lo = _clamp(int(values[i]))
        hi = _clamp(int(values[i + 1])) if i + 1 < n else 0
        out.append((hi << 4) | lo)
    return out


def unpack_int4(data: bytes, length: int) -> List[int]:
    """解包为 length 个 [0,15] 整数。"""
    out: List[int] = []
    for b in data:
        out.append(b & 0x0F)
        out.append((b >> 4) & 0x0F)
    return out[:length]


class QuantizedMatrix:
    """量化矩阵：int4 权重 + 逐组 scale/zero，支持反量化后做矩阵乘。

    W_real[i][j] = (W_q[i][j] - zero[g]) * scale[g]
    """

    def __init__(self, rows: int, cols: int, group_size: int = 128, sym: bool = True):
        self.rows = rows
        self.cols = cols
        self.group_size = group_size
        self.sym = sym
        self.q = bytearray()          # 4bit 打包权重
        self.scales: List[List[float]] = []
        self.zeros: List[List[int]] = []

    def quantize(self, W: Sequence[Sequence[float]]) -> "QuantizedMatrix":
        self.scales = []
        self.zeros = []
        flat: List[int] = []
        for i, row in enumerate(W):
            group_scales = []
            group_zeros = []
            for g in range(0, self.cols, self.group_size):
                chunk = row[g:g + self.group_size]
                mx = max(chunk)
                mn = min(chunk)
                if self.sym:
                    amax = max(abs(mx), abs(mn), 1e-9)
                    scale = amax / 7.0
                    zero = 8
                    group_scales.append(scale)
                    group_zeros.append(zero)
                    q = [_clamp(round(v / scale) + 8) for v in chunk]
                else:
                    rng = (mx - mn) or 1e-9
                    scale = rng / 15.0
                    zero = round(-mn / scale)
                    zero = _clamp(zero)
                    group_scales.append(scale)
                    group_zeros.append(zero)
                    q = [_clamp(round(v / scale) + zero) for v in chunk]
                flat.extend(q)
            self.scales.append(group_scales)
            self.zeros.append(group_zeros)
        self.q = pack_int4(flat)
        return self

    def dequantize(self) -> List[List[float]]:
        q = unpack_int4(bytes(self.q), self.rows * self.cols)
        out = []
        for i in range(self.rows):
            row = []
            gi = 0
            for j in range(self.cols):
                scale = self.scales[i][j // self.group_size]
                zero = self.zeros[i][j // self.group_size]
                row.append((q[i * self.cols + j] - zero) * scale)
            out.append(row)
        return out

    def nbytes(self) -> int:
        return len(self.q) + sum(len(s) for s in self.scales) * 4 + sum(len(z) for z in self.zeros) * 4
