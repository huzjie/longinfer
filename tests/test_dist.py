# -*- coding: utf-8 -*-
"""Dist 子系统测试。"""
from __future__ import annotations

import unittest




class TestDist(unittest.TestCase):
    def test_plan_sharding(self):
        from longinfer.dist.sharding import plan_sharding
        p = plan_sharding(256, 8, 1, 8, 2, 32000)
        self.assertEqual(len(p.expert_ids_per_rank), 8)
        self.assertEqual(len(p.vocab_shards_per_rank), 8)

    def test_dynamic_replication(self):
        from longinfer.dist.expert_parallel import dynamic_replication
        hot = [100, 5, 5, 5]
        reps = dynamic_replication(hot, 4, 8)
        self.assertEqual(sum(reps.values()), 8)
        self.assertGreater(reps[0], reps[1])

    def test_vocab_parallel(self):
        from longinfer.dist.vocab_parallel import VocabParallel
        vp = VocabParallel(1000, 2)
        self.assertEqual(vp.global_id(vp.shard_of(600), vp.local_id(600)), 600)

    def test_deployment_plan(self):
        from longinfer.dist.plan import build_deployment_plan
        p = build_deployment_plan(1600.0, 48.0, 8, 80.0, "w4a4")
        self.assertIn("B", p.summary())



if __name__ == "__main__":
    unittest.main()
