# -*- coding: utf-8 -*-
"""零拷贝通信抽象：封装跨卡传输，语义与实现解耦（可对接 NCCL/私有加速库）。"""
from __future__ import annotations

from typing import Any


class ZeroCopyComm:
    """零拷贝通信：在单机内用引用传递模拟；多机可替换为 RDMA/共享内存实现。"""

    def __init__(self, world_size: int, zero_copy: bool = True):
        self.world_size = world_size
        self.zero_copy = zero_copy
        self._buffers: dict[str, Any] = {}

    def register(self, name: str, obj: Any) -> None:
        self._buffers[name] = obj

    def read(self, name: str) -> Any:
        return self._buffers.get(name)

    def broadcast(self, name: str, src: int = 0) -> Any:
        """广播：返回共享引用（零拷贝），避免显式序列化。"""
        return self._buffers.get(name)

    def all_gather_tokens(self, local_counts: list) -> list:
        """跨 rank 汇总 token 计数，用于专家路由的全局负载统计。"""
        # 单机模拟：直接返回
        return list(local_counts)

    def barrier(self) -> None:
        return None
