# 稀疏 MoE 路由

## 原理

MoE（Mixture-of-Experts）把 Transformer 的 FFN 层替换成 N 个专家，每个 token 只被路由器分给 top_k 个专家。因此：

- **总参数**决定「知识容量」（如 LongCat-2.5 的 1.6T）
- **激活参数**决定「计算量」（如 48B）

## 关键点

1. **Top-K 路由**：`logits = W_gate @ x`，取 top_k 最大的专家，softmax 归一化做加权。
2. **辅助负载均衡**：不加约束时路由器会「偏心」，导致少数专家过载。aux-loss 惩罚专家使用频率的方差。
3. **容量因子**：每个专家单步最多处理 `capacity` 个 token，超出的丢到残差，防止显存峰值。

## 用法

```python
from longinfer.moe.moe_layer import MoELayer

layer = MoELayer(dim=16, num_experts=256, top_k=8)
y = layer.forward(x)             # 只走 8 个专家
print(layer.active_params())     # 激活参数量
print(layer.total_params())      # 总参数量
```

## 可复用要点

- 路由加 `jitter` 噪声可提升专家多样性（训练期）。
- 生产部署时按专家热度做**动态复制**（见 dist 章节）。
