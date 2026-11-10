# -*- coding: utf-8 -*-
"""离线 mock 模型：可复现的确定性生成，用于端到端验证与演示。"""
from __future__ import annotations

from typing import Sequence

from .sampler import Sampler
from ..config import Config


class MockModel:
    """确定性 mock：根据 prompt 哈希选择固定 logits，保证离线可复现。"""

    def __init__(self, config: Config):
        self.config = config
        self.sampler = Sampler(temperature=0.7, top_k=8, seed=42)

    def logits(self, prompt: str, n_vocab: int = 1024) -> Sequence[float]:
        import hashlib
        h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        return [((h >> (i % 16)) & 0xFF) / 255.0 for i in range(n_vocab)]

    def complete(self, prompt: str, max_new: int = 64) -> str:
        return f"[mock] 已处理「{prompt[:32]}…」并给出确定性应答"
