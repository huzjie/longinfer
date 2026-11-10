# -*- coding: utf-8 -*-
"""Vision 子系统测试。"""
from __future__ import annotations

import unittest




class TestVision(unittest.TestCase):
    def test_patchify(self):
        from longinfer.vision.patchify import patchify
        img = [[float(x + y) for x in range(32)] for y in range(32)]
        patches = patchify(img, 16)
        self.assertEqual(len(patches), 4)
        self.assertEqual(len(patches[0]), 256)

    def test_rvq_encode_decode(self):
        from longinfer.vision.rvq import ResidualVQ
        rvq = ResidualVQ(levels=4, codebook_size=64, code_dim=16)
        codes = rvq.encode([0.1] * 16)
        self.assertEqual(len(codes), 4)
        vec = rvq.decode(codes)
        self.assertEqual(len(vec), 16)

    def test_tokenizer(self):
        from longinfer.vision.image_loader import load_image
        from longinfer.vision.tokenizer import VisionTokenizer
        img = load_image("")
        tok = VisionTokenizer(patch_size=16, rvq_levels=4)
        tokens = tok.encode(img)
        self.assertGreater(len(tokens), 0)

    def test_codebook_nearest(self):
        from longinfer.vision.codebook import Codebook
        cb = Codebook(16, 8, seed=0)
        idx = cb.nearest(cb.codes[3])
        self.assertEqual(idx, 3)



if __name__ == "__main__":
    unittest.main()
