# 视觉 RVQ 离散化

## 原理

LongCat-2.5 不外挂 ViT，而是把图像信号统一编码为**多层级离散 token**：

1. **patchify**：把图像切成 p×p 块。
2. **特征投影**：patch 向量线性投影到 code_dim。
3. **残差向量量化（RVQ）**：逐级量化残差——第 0 级量化原始向量，后续层级量化上一级的残差，逐级补足细节。
4. 视觉码本与主干联合训练，视觉/文本共享同一套参数、同一语义空间。

## 用法

```python
from longinfer.vision.tokenizer import VisionTokenizer
tokens = VisionTokenizer(patch_size=16, rvq_levels=4).encode(image)
# tokens[i] = [c0, c1, c2, c3] 每 patch 的 4 级码下标
```

## 可复用要点

- RVQ 层级越多细节越丰富，但 token 数成倍增加；4 层是常见权衡点。
- 码本维度与主干 hidden dim 对齐可避免额外投影层。
