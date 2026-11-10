# -*- coding: utf-8 -*-
"""连续批处理调度：把多请求动态组批，提升吞吐（continuous batching）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Batch:
    seq_ids: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)


class ContinuousBatchingScheduler:
    """维护一个运行中的请求池，prefill 与 decode 交替，新请求可随时加入。"""

    def __init__(self, max_batch_size: int = 256):
        self.max_batch_size = max_batch_size
        self.pool: List[dict] = []
        self._seq_counter = 0

    def add(self, prompt_tokens: List[int], max_new: int = 256) -> int:
        seq_id = self._seq_counter
        self._seq_counter += 1
        self.pool.append({"id": seq_id, "prompt": prompt_tokens, "generated": [], "max_new": max_new, "done": False})
        return seq_id

    def schedule(self) -> Batch:
        active = [s for s in self.pool if not s["done"]]
        active = active[:self.max_batch_size]
        batch = Batch()
        for s in active:
            batch.seq_ids.append(s["id"])
            # prefill 阶段消费 prompt，decode 阶段消费已生成 token
            if s["generated"]:
                batch.tokens.append(s["generated"][-1])
            else:
                batch.tokens.extend(s["prompt"])
        return batch

    def update(self, seq_id: int, token: int, done: bool = False) -> None:
        for s in self.pool:
            if s["id"] == seq_id:
                s["generated"].append(token)
                if done or len(s["generated"]) >= s["max_new"]:
                    s["done"] = True
                return

    def finished(self, seq_id: int) -> Optional[List[int]]:
        for s in self.pool:
            if s["id"] == seq_id and s["done"]:
                return s["generated"]
        return None

    def stats(self) -> dict:
        total = len(self.pool)
        active = sum(1 for s in self.pool if not s["done"])
        return {"total": total, "active": active}
