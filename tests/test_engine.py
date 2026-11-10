# -*- coding: utf-8 -*-
"""Engine 子系统测试。"""
from __future__ import annotations

import unittest




class TestEngine(unittest.TestCase):
    def test_simple_tokenizer(self):
        from longinfer.engine.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer()
        ids = tok.encode("你好")
        self.assertEqual(tok.decode(ids), "你好")

    def test_greedy(self):
        from longinfer.engine.sampler import greedy
        self.assertEqual(greedy([0.1, 0.9, 0.5]), 1)

    def test_sample_range(self):
        from longinfer.engine.sampler import sample
        for _ in range(50):
            i = sample([0.25, 0.25, 0.25, 0.25], temperature=1.0)
            self.assertIn(i, range(4))

    def test_paged_kv_cache(self):
        from longinfer.engine.kv_cache import PagedKVCache
        c = PagedKVCache(page_size=4)
        for _ in range(10):
            c.append([0.0] * 4, [0.0] * 4)
        self.assertEqual(c.length(), 10)

    def test_scheduler(self):
        from longinfer.engine.scheduler import ContinuousBatchingScheduler
        s = ContinuousBatchingScheduler(8)
        sid = s.add([1, 2, 3], max_new=2)
        s.update(sid, 5)
        s.update(sid, 6)
        self.assertIsNotNone(s.finished(sid))



if __name__ == "__main__":
    unittest.main()
