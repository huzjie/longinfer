# SDK

## Python

```python
from longinfer.sdk import LonginferClient
c = LonginferClient("http://127.0.0.1:8000")
print(c.health())
print(c.chat("你好"))
print(c.agent_run("测试"))
```

## TypeScript

见 `sdk/typescript/index.ts`，导出 `LonginferClient`。
