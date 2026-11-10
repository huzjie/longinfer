# -*- coding: utf-8 -*-
"""网页抓取并抽取文本"""
from __future__ import annotations


class WebTool:
    name = "web"
    description = "网页抓取并抽取文本"
    parameters = {"url": "str"}

    def run(self, **kwargs):
        import re, urllib.request
        url = kwargs.get("url", "")
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                html = r.read().decode("utf-8", errors="replace")
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:2000]
        except Exception as e:  # noqa
            return f"抓取失败: {e}"

