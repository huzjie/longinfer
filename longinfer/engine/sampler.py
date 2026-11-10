# -*- coding: utf-8 -*-
"""采样策略：贪心、温度采样、Top-K、Top-P、重复惩罚。"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence

from ..utils.tensor import softmax


def greedy(logits: Sequence[float]) -> int:
    return max(range(len(logits)), key=lambda i: logits[i])


def top_k_filter(logits: Sequence[float], k: int) -> List[float]:
    if k <= 0 or k >= len(logits):
        return list(logits)
    idx = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:k]
    out = [-1e18] * len(logits)
    for i in idx:
        out[i] = logits[i]
    return out


def top_p_filter(logits: Sequence[float], p: float) -> List[float]:
    probs = softmax(logits)
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    out = [-1e18] * len(logits)
    cum = 0.0
    for i in order:
        out[i] = logits[i]
        cum += probs[i]
        if cum >= p:
            break
    return out


def sample(logits: Sequence[float], temperature: float = 1.0, top_k: int = 0,
           top_p: float = 1.0, seed: Optional[int] = None) -> int:
    logits = list(logits)
    if top_k > 0:
        logits = top_k_filter(logits, top_k)
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    if temperature <= 1e-9:
        return greedy(logits)
    probs = softmax(logits, temperature)
    rng = random.Random(seed)
    r = rng.random()
    cum = 0.0
    for i, pr in enumerate(probs):
        cum += pr
        if r <= cum:
            return i
    return len(probs) - 1


class Sampler:
    def __init__(self, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, seed: Optional[int] = None):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed

    def __call__(self, logits: Sequence[float]) -> int:
        return sample(logits, self.temperature, self.top_k, self.top_p, self.seed)
