# 内置工具

| 工具 | 用途 |
|---|---|
| shell | 只读命令执行（沙箱约束） |
| files | 文件读写 |
| http | HTTP 请求 |
| search | 检索 |
| calc | 安全计算（白名单字符） |
| browser | 网页获取 |
| time | 当前时间 |
| web | 网页文本抽取 |

```python
from longinfer.tools import build_default_registry
reg = build_default_registry()
print(reg.describe())
```
