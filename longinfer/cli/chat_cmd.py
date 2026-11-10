# -*- coding: utf-8 -*-
"""交互式对话。"""
from __future__ import annotations

from ..logging import get_logger

log = get_logger("longinfer.chat")


def chat(config, args) -> int:
    from ..api.inference import build_provider
    provider = build_provider(config)
    print("longinfer 对话（输入 exit 退出）")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ("exit", "quit", "q"):
            break
        if not line:
            continue
        print("ai>", provider.complete(line, 256))
    return 0
