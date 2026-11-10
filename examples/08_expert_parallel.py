# -*- coding: utf-8 -*-
"""示例 8：专家并行与动态复制。"""
from longinfer.dist.expert_parallel import dynamic_replication

hot = [90, 40, 20, 10, 5, 3, 2, 1]
reps = dynamic_replication(hot, num_ranks=8, total_capacity=12)
for e, r in sorted(reps.items(), key=lambda x: -x[1]):
    print(f"专家 {e}: {r} 副本")
