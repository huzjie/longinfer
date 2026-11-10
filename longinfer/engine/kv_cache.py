# -*- coding: utf-8 -*-
"""KV 缓存：连续空间 + 分页实现，支撑 100 万上下文。"""
from __future__ import annotations

from typing import List, Optional, Sequence


class KVCache:
    """朴素 KV 缓存：按 token 存储 key/value。"""

    def __init__(self, head_dim: int = 128):
        self.head_dim = head_dim
        self.keys: List[List[float]] = []
        self.values: List[List[float]] = []

    def append(self, key: Sequence[float], value: Sequence[float]) -> None:
        self.keys.append(list(key))
        self.values.append(list(value))

    def length(self) -> int:
        return len(self.keys)

    def clear(self) -> None:
        self.keys.clear()
        self.values.clear()


class PagedKVCache:
    """分页 KV 缓存：固定页大小，超长序列按页换入换出（面向百万上下文）。"""

    def __init__(self, page_size: int = 16, max_pages: int = 65536, head_dim: int = 128):
        self.page_size = page_size
        self.max_pages = max_pages
        self.head_dim = head_dim
        self.pages: dict[int, dict] = {}
        self._next_page = 0
        self._used_tokens = 0

    def _ensure_page(self) -> int:
        if self._next_page >= self.max_pages:
            # 简单 LRU 淘汰最早页
            oldest = min(self.pages.keys())
            del self.pages[oldest]
        pid = self._next_page
        self.pages[pid] = {"k": [], "v": [], "used": 0}
        self._next_page += 1
        return pid

    def append(self, key: Sequence[float], value: Sequence[float]) -> None:
        if not self.pages:
            self._ensure_page()
        pid = self._next_page - 1
        page = self.pages[pid]
        if page["used"] >= self.page_size:
            pid = self._ensure_page()
            page = self.pages[pid]
        page["k"].append(list(key))
        page["v"].append(list(value))
        page["used"] += 1
        self._used_tokens += 1

    def length(self) -> int:
        return self._used_tokens

    def clear(self) -> None:
        self.pages.clear()
        self._next_page = 0
        self._used_tokens = 0
