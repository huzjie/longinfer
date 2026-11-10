# -*- coding: utf-8 -*-
"""Tools 子系统测试。"""
from __future__ import annotations

import unittest




class TestTools(unittest.TestCase):
    def test_build_registry(self):
        from longinfer.tools import build_default_registry
        reg = build_default_registry()
        self.assertIn("calc", reg.names())
        self.assertIn("shell", reg.names())

    def test_calc(self):
        from longinfer.tools import build_default_registry
        reg = build_default_registry()
        out = reg.get("calc").run(expr="(2+3)*4")
        self.assertEqual(out, "20")

    def test_calc_rejects_unsafe(self):
        from longinfer.tools import build_default_registry
        reg = build_default_registry()
        out = reg.get("calc").run(expr="__import__('os')")
        self.assertIn("不允许", out)



if __name__ == "__main__":
    unittest.main()
