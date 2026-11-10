# -*- coding: utf-8 -*-
"""发起 HTTP GET/POST 请求"""
from __future__ import annotations


class HttpTool:
    name = "http"
    description = "发起 HTTP GET/POST 请求"
    parameters = {"url": "str", "method": "str = GET"}

    def run(self, **kwargs):
        import json, urllib.request
        url = kwargs["url"]
        method = kwargs.get("method", "GET").upper()
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read().decode("utf-8", errors="replace")
            return body[:2000]

