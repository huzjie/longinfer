# -*- coding: utf-8 -*-
"""获取当前时间"""
from __future__ import annotations


class TimeTool:
    name = "time"
    description = "获取当前时间"
    parameters = {}

    def run(self, **kwargs):
        import time
        return time.strftime("%Y-%m-%d %H:%M:%S %Z")

