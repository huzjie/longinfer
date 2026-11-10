# -*- coding: utf-8 -*-
"""示例 5：分布式部署计划。"""
from longinfer.dist.plan import build_deployment_plan

for quant in ("w4a4", "bf16"):
    p = build_deployment_plan(1600.0, 48.0, 8, per_card_gb=80.0, quant=quant)
    print(f"--- {quant} ---")
    print(p.summary())
