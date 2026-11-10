# -*- coding: utf-8 -*-
"""FP8（E4M3）量化：用于 KV 缓存与激活的 8bit 浮点压缩。"""
from __future__ import annotations

from typing import List, Sequence

# E4M3：1 符号 + 4 指数 + 3 尾数
_MAX_E4M3 = 448.0


def quantize_fp8(values: Sequence[float]) -> List[int]:
    """量化到 E4M3 的整数表示（0-255）。此处用简化的 2 的幂对齐。"""
    out: List[int] = []
    for v in values:
        av = abs(v)
        if av == 0:
            out.append(0)
            continue
        import math
        exp = math.floor(math.log2(av))
        mant = av / (2.0 ** exp)
        # 3bit 尾数 -> 8 档
        m = round((mant - 1.0) * 8.0)
        m = max(0, min(7, m))
        # 编码为 8bit：符号(1) + 指数(4, 偏置 7) + 尾数(3)
        sign = 1 if v < 0 else 0
        e = max(0, min(15, exp + 7))
        code = (sign << 7) | (e << 3) | m
        out.append(code)
    return out


def dequantize_fp8(codes: Sequence[int]) -> List[float]:
    out = []
    for c in codes:
        sign = -1.0 if (c >> 7) & 1 else 1.0
        e = (c >> 3) & 0xF
        m = c & 0x7
        exp = e - 7
        mant = 1.0 + m / 8.0
        out.append(sign * mant * (2.0 ** exp))
    return out
