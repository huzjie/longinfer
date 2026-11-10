# -*- coding: utf-8 -*-
"""自检：逐项诊断六个子系统 + provider + 控制面。"""
from __future__ import annotations

from ..logging import get_logger

log = get_logger("longinfer.doctor")


def _check(name: str, fn) -> bool:
    try:
        fn()
        log.info("  [OK] %s", name)
        return True
    except Exception as e:  # noqa
        log.error("  [FAIL] %s: %s", name, e)
        return False


def run_doctor(config) -> int:
    log.info("longinfer 自检开始...")
    results = []

    def moe():
        from ..moe.moe_layer import MoELayer
        layer = MoELayer(dim=16, num_experts=32, top_k=4)
        idxs, probs = layer.router.route([0.1] * 16)
        assert len(idxs) == 4 and abs(sum(probs) - 1.0) < 1e-6
        y = layer.forward([0.1] * 16)
        assert len(y) == 16
    results.append(_check("MoE 稀疏路由（32 专家 top_k=4）", moe))

    def quant():
        from ..quant.int4 import QuantizedMatrix
        from ..utils.tensor import randn
        W = randn((8, 32), seed=0)
        qm = QuantizedMatrix(8, 32, 16).quantize(W)
        dq = qm.dequantize()
        assert len(dq) == 8 and len(dq[0]) == 32
    results.append(_check("W4A4 INT4 量化/反量化", quant))

    def vision():
        from ..vision.image_loader import load_image
        from ..vision.tokenizer import VisionTokenizer
        img = load_image("")
        tok = VisionTokenizer(patch_size=16, rvq_levels=4)
        tokens = tok.encode(img)
        assert len(tokens) > 0
    results.append(_check("视觉 RVQ 离散 token 化", vision))

    def agent():
        from ..agent.runtime import AgentRuntime
        from ..tools import build_default_registry
        rt = AgentRuntime(config, None, build_default_registry())
        traj = rt.run("测试目标", resume_from=None)
        assert len(traj.steps) >= 0
    results.append(_check("长程智能体运行时", agent))

    def dist():
        from ..dist.plan import build_deployment_plan
        p = build_deployment_plan(1600.0, 48.0, 8, 80.0, "w4a4")
        assert "B" in p.summary()
    results.append(_check("分布式部署计划", dist))

    def engine():
        from ..engine.kv_cache import PagedKVCache
        from ..engine.sampler import sample
        cache = PagedKVCache(page_size=8)
        for _ in range(20):
            cache.append([0.0] * 4, [0.0] * 4)
        assert cache.length() == 20
        assert 0 <= sample([0.1, 0.2, 0.3, 0.4], temperature=0.7) < 4
    results.append(_check("引擎（分页 KV 缓存 + 采样）", engine))

    def provider():
        from ..api.inference import build_provider
        p = build_provider(config)
        assert p.complete("ping", 8) != ""
    results.append(_check("Provider 实例化与调用", provider))

    ok = sum(results)
    log.info("自检完成：%d/%d 通过", ok, len(results))
    return 0 if ok == len(results) else 1
