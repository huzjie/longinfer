# -*- coding: utf-8 -*-
"""Config 子系统测试。"""
from __future__ import annotations

import unittest




class TestConfig(unittest.TestCase):
    def test_defaults(self):
        from longinfer.config import Config
        c = Config()
        self.assertEqual(c["provider"], "mock")
        self.assertEqual(c["moe"]["top_k"], 8)

    def test_mini_yaml(self):
        from longinfer._yaml import load
        data = load("a: 1\nb:\n  c: hello\n  d: [1, 2, 3]\n")
        self.assertEqual(data["a"], 1)
        self.assertEqual(data["b"]["c"], "hello")
        self.assertEqual(data["b"]["d"], [1, 2, 3])



if __name__ == "__main__":
    unittest.main()
