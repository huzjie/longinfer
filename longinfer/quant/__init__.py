# -*- coding: utf-8 -*-
"""W4A4 量化：权重与激活同压 4bit，配合原生 INT4 矩阵乘。

对应 LongCat-2.5 的「W4A4 方案 + 原生 INT4 矩阵乘法单元」。
"""
from .int4 import pack_int4, unpack_int4, QuantizedMatrix
from .gptq import gptq_quantize
from .awq import awq_quantize, protect_outliers
from .dequant import dequantize_matrix
from .calibration import calibrate_activations
from .fp8 import quantize_fp8, dequantize_fp8

__all__ = [
    "pack_int4", "unpack_int4", "QuantizedMatrix",
    "gptq_quantize", "awq_quantize", "protect_outliers",
    "dequantize_matrix", "calibrate_activations",
    "quantize_fp8", "dequantize_fp8",
]
