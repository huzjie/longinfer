# -*- coding: utf-8 -*-
"""Partial rollout：长任务可中断，保存检查点后从断点续跑。"""
from __future__ import annotations

from .trajectory import Trajectory


class PartialRollout:
    """支持中断续跑的 rollout 调度：按 step 粒度推进，周期保存检查点。"""

    def __init__(self, checkpoint_every: int = 10, oversample_by_pass_rate: bool = True):
        self.checkpoint_every = checkpoint_every
        self.oversample_by_pass_rate = oversample_by_pass_rate

    def should_checkpoint(self, step: int) -> bool:
        return step > 0 and step % self.checkpoint_every == 0

    def adjust_oversampling(self, pass_rate: float) -> int:
        """按任务通过率动态调整过采样：通过率低的轨迹多采样，剔除全对/全错样本。"""
        if not self.oversample_by_pass_rate:
            return 1
        if pass_rate >= 1.0 or pass_rate <= 0.0:
            return 0   # 全对/全错：低信息量，剔除
        # 通过率越接近 0.5 信息量越大，采样权重越高
        return max(1, int(4 * min(pass_rate, 1 - pass_rate) * 2))
