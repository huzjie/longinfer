# -*- coding: utf-8 -*-
"""示例 2：W4A4 量化与误差评估。"""
from longinfer.quant.int4 import QuantizedMatrix
from longinfer.utils.tensor import randn

W = randn((16, 64), seed=0)
qm = QuantizedMatrix(16, 64, group_size=16).quantize(W)
dq = qm.dequantize()
max_err = max(abs(a - b) for ra, rb in zip(W, dq) for a, b in zip(ra, rb))
print("FP32 字节:", 16 * 64 * 4)
print("量化后字节:", qm.nbytes())
print("压缩比:", round((16 * 64 * 4) / qm.nbytes(), 2))
print("最大误差:", round(max_err, 4))
