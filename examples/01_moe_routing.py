# -*- coding: utf-8 -*-
"""示例 1：稀疏 MoE 路由 + 激活/总参数量。"""
from longinfer.moe.moe_layer import MoELayer

layer = MoELayer(dim=64, num_experts=256, top_k=8)
x = [0.01] * 64
idxs, probs = layer.router.route(x)
print("选中专家:", idxs)
print("路由权重:", [round(p, 4) for p in probs])
print("激活参数:", layer.active_params())
print("总参数:", layer.total_params())
print("稀疏比:", round(layer.total_params() / layer.active_params(), 1))
