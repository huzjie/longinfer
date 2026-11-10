# -*- coding: utf-8 -*-
"""安全计算数学表达式"""
from __future__ import annotations


class CalcTool:
    name = "calc"
    description = "安全计算数学表达式"
    parameters = {"expr": "str"}

    def run(self, **kwargs):
        expr = kwargs.get("expr", "")
        allowed = set("0123456789+-*/().% ")
        if any(c not in allowed for c in expr):
            return "表达式含不允许的字符"
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception as e:  # noqa
            return f"计算失败: {e}"

