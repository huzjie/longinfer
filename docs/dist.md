# 分布式部署

## 并行维度

| 维度 | 作用 | 对应 LongCat-2.5 |
|---|---|---|
| 张量并行 TP | 切分单层权重 | 单卡显存 < 80GB 时分片到超节点多卡 |
| 专家并行 EP | 把专家分布到多卡 | MoonEP 动态专家复制 |
| 词表并行 VP | 切分大词表 | 混合词表并行 |

## 动态专家复制

按专家访问热度分配副本：热门专家多副本、冷门专家合并，显存利用率最优。

## 用法

```python
from longinfer.dist.plan import build_deployment_plan
plan = build_deployment_plan(1600.0, 48.0, 8, per_card_gb=80.0, quant="w4a4")
print(plan.summary())
```
