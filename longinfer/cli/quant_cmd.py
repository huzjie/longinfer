# -*- coding: utf-8 -*-
"""W4A4 量化演示：展示量化误差与体积压缩。"""
from __future__ import annotations

from ..logging import get_logger

log = get_logger("longinfer.quant")


def quant(config, args) -> int:
    from ..quant.int4 import QuantizedMatrix
    from ..utils.tensor import randn
    W = randn((args.rows, args.cols), seed=0)
    qm = QuantizedMatrix(args.rows, args.cols, group_size=args.cols).quantize(W)
    dq = qm.dequantize()
    max_err = max(abs(a - b) for ra, rb in zip(W, dq) for a, b in zip(ra, rb))
    fp32_bytes = args.rows * args.cols * 4
    log.info("方法=%s 矩阵=%dx%d 量化后=%d 字节（FP32=%d 字节，压缩 %.1f%%）",
             args.method, args.rows, args.cols, qm.nbytes(), fp32_bytes, (1 - qm.nbytes() / fp32_bytes) * 100)
    log.info("最大量化误差 = %.4f", max_err)
    return 0
