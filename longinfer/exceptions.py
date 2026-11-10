# -*- coding: utf-8 -*-
"""统一异常体系。"""
from __future__ import annotations


class LonginferError(Exception):
    """所有 longinfer 异常的基类。"""


class ConfigError(LonginferError):
    """配置加载/校验失败。"""


class ProviderError(LonginferError):
    """推理后端调用失败。"""


class QuantizationError(LonginferError):
    """量化/反量化失败。"""


class RoutingError(LonginferError):
    """MoE 路由异常（专家越界、容量溢出等）。"""


class VisionError(LonginferError):
    """视觉编码/RVQ 量化失败。"""


class AgentError(LonginferError):
    """智能体运行时异常。"""


class CheckpointError(LonginferError):
    """检查点读写异常。"""


class DistError(LonginferError):
    """分布式部署/分片异常。"""


class SamplingError(LonginferError):
    """采样异常。"""
