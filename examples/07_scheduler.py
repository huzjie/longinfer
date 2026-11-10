# -*- coding: utf-8 -*-
"""示例 7：连续批处理调度。"""
from longinfer.engine.scheduler import ContinuousBatchingScheduler

s = ContinuousBatchingScheduler(max_batch_size=8)
a = s.add([1, 2, 3], max_new=3)
b = s.add([4, 5], max_new=2)
s.update(a, 10)
s.update(a, 11)
s.update(a, 12)
s.update(b, 20)
s.update(b, 21)
print("stats:", s.stats())
print("seq a 完成:", s.finished(a))
