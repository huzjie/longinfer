# -*- coding: utf-8 -*-
"""分布式推理：面向显存/带宽受限的国产算力的分片与并行方案。

对应 LongCat-2.5 的「分片超节点多卡 + 混合词表并行 + MoonEP 动态专家复制 + 零拷贝通信」。
"""
from .topology import Topology, Device
from .sharding import ShardingPlan, plan_sharding
from .expert_parallel import ExpertParallel, dynamic_replication
from .vocab_parallel import VocabParallel
from .comm import ZeroCopyComm
from .plan import DeploymentPlan, build_deployment_plan

__all__ = [
    "Topology", "Device", "ShardingPlan", "plan_sharding",
    "ExpertParallel", "dynamic_replication", "VocabParallel",
    "ZeroCopyComm", "DeploymentPlan", "build_deployment_plan",
]
