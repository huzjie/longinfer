# -*- coding: utf-8 -*-
"""Providers 子系统测试。"""
from __future__ import annotations

import unittest




class TestProviders(unittest.TestCase):
    def test_mock(self):
        from longinfer.providers.mock import MockProvider
        p = MockProvider()
        self.assertNotEqual(p.complete("ping", 8), "")

    def test_factory_mock(self):
        from longinfer.config import Config
        from longinfer.api.inference import build_provider
        p = build_provider(Config())
        self.assertIsNotNone(p)

    def test_factory_longcat(self):
        from longinfer.config import Config
        from longinfer.api.inference import build_provider
        c = Config({"provider": "longcat", "model": "LongCat-2.5", "base_url": "http://127.0.0.1/v1"})
        p = build_provider(c)
        self.assertEqual(p.name, "longcat")



if __name__ == "__main__":
    unittest.main()
