# -*- coding: utf-8 -*-
"""离线完整跑一遍：provider -> 采样 -> 应答。"""
from __future__ import annotations

from ..logging import get_logger

log = get_logger("longinfer.run")


def run(config, args) -> int:
    from ..api.inference import build_provider
    provider = build_provider(config)
    log.info("provider=%s model=%s", config["provider"], config["model"])
    out = provider.complete(args.prompt or "解释 MoE 稀疏路由", max_new=128)
    print(out)
    return 0
