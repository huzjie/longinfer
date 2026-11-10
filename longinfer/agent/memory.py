# -*- coding: utf-8 -*-
"""智能体记忆：短期滑动窗口 + 长期键值（默认内存实现）。"""
from __future__ import annotations

from collections import OrderedDict


class AgentMemory:
    def __init__(self, short_capacity: int = 20, long_capacity: int = 1000):
        self.short_capacity = short_capacity
        self.long_capacity = long_capacity
        self.short = OrderedDict()   # 滑动窗口
        self.long: dict = {}         # 长期键值

    def add_observation(self, text: str) -> None:
        self.short[text] = True
        while len(self.short) > self.short_capacity:
            self.short.popitem(last=False)

    def remember(self, key: str, value) -> None:
        self.long[key] = value

    def recall(self, key: str, default=None):
        return self.long.get(key, default)

    def context(self) -> list:
        return list(self.short.keys())
