# -*- coding: utf-8 -*-
"""Utils 子系统测试。"""
from __future__ import annotations

import unittest




class TestUtils(unittest.TestCase):
    def test_mat_vec(self):
        from longinfer.utils.tensor import mat_vec
        self.assertEqual(mat_vec([[1, 2], [3, 4]], [1, 1]), [3, 7])

    def test_softmax(self):
        from longinfer.utils.tensor import softmax
        p = softmax([1, 1, 1])
        self.assertAlmostEqual(sum(p), 1.0, places=5)

    def test_topk(self):
        from longinfer.utils.tensor import topk_indices
        self.assertEqual(topk_indices([1, 5, 3, 9], 2), [3, 1])

    def test_hash(self):
        from longinfer.utils.hash import sha256_bytes, content_hash
        self.assertEqual(len(sha256_bytes(b"x")), 64)
        self.assertEqual(len(content_hash({"a": 1})), 16)



if __name__ == "__main__":
    unittest.main()
