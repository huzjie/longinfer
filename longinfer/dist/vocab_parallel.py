# -*- coding: utf-8 -*-
"""混合词表并行：大词表按 VP 切分，降低显存占用并提升长序列吞吐。"""
from __future__ import annotations


class VocabParallel:
    def __init__(self, vocab_size: int, vocab_parallel: int = 2):
        self.vocab_size = vocab_size
        self.vp = vocab_parallel
        self.shard_size = (vocab_size + self.vp - 1) // self.vp

    def shard_of(self, token_id: int) -> int:
        return token_id // self.shard_size

    def local_id(self, token_id: int) -> int:
        return token_id % self.shard_size

    def global_id(self, shard: int, local: int) -> int:
        return shard * self.shard_size + local
