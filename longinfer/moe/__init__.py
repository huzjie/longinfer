# -*- coding: utf-8 -*-
"""稀疏 Mixture-of-Experts：路由 + 专家 + 负载均衡。

对应 LongCat-2.5 的「1.6T 总参、48B 激活」：每个 token 只被路由器分给 top_k 个专家。
"""
from .routing import Router
from .experts import Expert, FFNExpert
from .moe_layer import MoELayer
from .load_balancer import LoadBalancer, aux_load_balance_loss

__all__ = ["Router", "Expert", "FFNExpert", "MoELayer", "LoadBalancer", "aux_load_balance_loss"]
