# -*- coding: utf-8 -*-
"""微基准：路由/量化/采样吞吐。"""
from __future__ import annotations

import time

from ..logging import get_logger

log = get_logger("longinfer.bench")


def benchmark(config, args) -> int:
    from ..moe.moe_layer import MoELayer
    from ..engine.sampler import sample
    n = args.tokens
    layer = MoELayer(dim=16, num_experts=64, top_k=8)
    vec = [0.01] * 16
    t0 = time.perf_counter()
    for _ in range(n):
        layer.router.route(vec)
    route_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n):
        sample([i / 100.0 for i in range(100)], temperature=0.7)
    sample_sec = time.perf_counter() - t0
    log.info("MoE 路由：%d 次 / %.3fs = %.0f token/s", n, route_sec, n / route_sec)
    log.info("采样：%d 次 / %.3fs = %.0f token/s", n, sample_sec, n / sample_sec)
    return 0
