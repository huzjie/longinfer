# W4A4 量化

## 原理

权重与激活同时压到 4bit（W4A4），配合原生 INT4 矩阵乘单元，1.8TB 权重可压到约 1/4 体积。

`W_real = (W_q - zero) * scale`，其中 scale 按**组**（group_size）或按**通道**计算。

## 支持的量化方法

| 方法 | 说明 |
|---|---|
| `w4a4` | 权重/激活同 4bit，逐组 scale/zero |
| `gptq` | 逐列最小二乘误差补偿 |
| `awq` | 按激活幅值保护重要通道 |
| `fp8` | E4M3 8bit 浮点，用于 KV 缓存/激活 |

## 用法

```python
from longinfer.quant.int4 import QuantizedMatrix
qm = QuantizedMatrix(rows, cols, group_size=128).quantize(W)
W_dequant = qm.dequantize()
```

## 可复用要点

- 保护离群通道（激活幅值大的）可显著降低量化误差——这就是 AWQ 的核心。
- 量化后先算 `max_abs_error` 再上线，误差 >0.2 建议加细 group_size。
