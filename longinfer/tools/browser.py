# -*- coding: utf-8 -*-
"""打开网页并返回正文摘要（简化实现）"""
from __future__ import annotations


class BrowserTool:
    name = "browser"
    description = "打开网页并返回正文摘要（简化实现）"
    parameters = {"url": "str"}

    def run(self, **kwargs):
        import urllib.request
        url = kwargs.get("url", "")
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                html = r.read().decode("utf-8", errors="replace")
                return f"[browser] 已获取 {len(html)} 字节"
        except Exception as e:  # noqa
            return f"浏览器打开失败: {e}"

