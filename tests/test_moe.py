# -*- coding: utf-8 -*-
"""MoE 子系统测试。"""
from __future__ import annotations

import unittest




class TestMoe(unittest.TestCase):
    def test_router_topk(self):
        from longinfer.moe.moe_layer import MoELayer
        layer = MoELayer(dim=16, num_experts=32, top_k=4)
        idxs, probs = layer.router.route([0.1] * 16)
        self.assertEqual(len(idxs), 4)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_forward_shape(self):
        from longinfer.moe.moe_layer import MoELayer
        layer = MoELayer(dim=8, num_experts=8, top_k=2)
        y = layer.forward([0.1] * 8)
        self.assertEqual(len(y), 8)

    def test_active_less_than_total(self):
        from longinfer.moe.moe_layer import MoELayer
        layer = MoELayer(dim=8, num_experts=16, top_k=2)
        self.assertLess(layer.active_params(), layer.total_params())

    def test_aux_loss(self):
        from longinfer.moe.load_balancer import aux_load_balance_loss
        routes = [[0, 1], [0, 1], [2, 3]]
        loss = aux_load_balance_loss(routes, 4)
        self.assertGreaterEqual(loss, 0.0)

    def test_capacity_enforce(self):
        from longinfer.moe.load_balancer import LoadBalancer
        lb = LoadBalancer(4, capacity_factor=1.0)
        routes = [[0], [0], [0], [0]]
        kept, overflow = lb.enforce(routes, capacity=2)
        self.assertGreaterEqual(overflow, 0)



if __name__ == "__main__":
    unittest.main()
