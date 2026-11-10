# -*- coding: utf-8 -*-
"""激活校准：从代表性样本统计激活分布，用于 AWQ 保护与动态 scale。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence


def calibrate_activations(samples: Sequence[Sequence[float]]) -> dict:
    """输入 shape: [样本数, 通道数]，输出每通道的均值/最大幅值/标准差。"""
    if not samples:
        return {"mean": [], "amax": [], "std": []}
    n_ch = len(samples[0])
    mean = [0.0] * n_ch
    amax = [0.0] * n_ch
    for row in samples:
        for c in range(n_ch):
            mean[c] += row[c]
            amax[c] = max(amax[c], abs(row[c]))
    mean = [m / len(samples) for m in mean]
    var = [0.0] * n_ch
    for row in samples:
        for c in range(n_ch):
            var[c] += (row[c] - mean[c]) ** 2
    std = [(v / len(samples)) ** 0.5 for v in var]
    return {"mean": mean, "amax": amax, "std": std}


def load_samples_jsonl(path: str, field: str = "activations", limit: int = 128) -> List[Sequence[float]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if field in obj and isinstance(obj[field], list):
                out.append(obj[field])
    return out
