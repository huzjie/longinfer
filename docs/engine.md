# 推理引擎内核

- **采样器**：贪心 / 温度 / Top-K / Top-P / 重复惩罚。
- **KV 缓存**：连续空间 + 分页实现，支撑百万上下文。
- **连续批处理调度**：prefill 与 decode 交替，新请求随时加入。

```python
from longinfer.engine.kv_cache import PagedKVCache
cache = PagedKVCache(page_size=16)
cache.append(key, value)
```
