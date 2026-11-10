# -*- coding: utf-8 -*-
"""Quant 子系统测试。"""
from __future__ import annotations

import unittest




class TestQuant(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        from longinfer.quant.int4 import pack_int4, unpack_int4
        vals = [1, 15, 0, 7, 3, 15]
        packed = pack_int4(vals)
        unpacked = unpack_int4(bytes(packed), len(vals))
        self.assertEqual(unpacked, vals)

    def test_matrix_quant_dequant(self):
        from longinfer.quant.int4 import QuantizedMatrix
        from longinfer.utils.tensor import randn
        W = randn((4, 16), seed=0)
        qm = QuantizedMatrix(4, 16, group_size=16).quantize(W)
        dq = qm.dequantize()
        self.assertEqual(len(dq), 4)
        self.assertEqual(len(dq[0]), 16)
        self.assertLess(qm.nbytes(), 4 * 16 * 4)

    def test_fp8_roundtrip(self):
        from longinfer.quant.fp8 import quantize_fp8, dequantize_fp8
        vals = [0.5, -0.25, 1.0, 0.125]
        codes = quantize_fp8(vals)
        back = dequantize_fp8(codes)
        self.assertEqual(len(back), 4)

    def test_awq_protect(self):
        from longinfer.quant.awq import protect_outliers
        act = [0.1, 100.0, 0.2, 0.3, 50.0]
        protected = protect_outliers(act, top_ratio=0.2)
        self.assertIn(1, protected)



if __name__ == "__main__":
    unittest.main()
