# -*- coding: utf-8 -*-
"""示例 6：REST API 调用（先启动 longinfer serve）。"""
from longinfer.sdk import LonginferClient

c = LonginferClient("http://127.0.0.1:8000")
print("health:", c.health())
print("chat:", c.chat("你好")["text"])
