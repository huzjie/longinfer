# -*- coding: utf-8 -*-
"""读写工作区文件"""
from __future__ import annotations


class FilesTool:
    name = "files"
    description = "读写工作区文件"
    parameters = {"path": "str", "content": "str = None"}

    def run(self, **kwargs):
        from pathlib import Path
        p = Path(kwargs["path"])
        content = kwargs.get("content")
        if content is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"已写入 {p}"
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")[:2000]
        return f"文件不存在: {p}"

