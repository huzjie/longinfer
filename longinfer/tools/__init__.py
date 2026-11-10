# -*- coding: utf-8 -*-
"""内置智能体工具集合，可注册到 AgentRuntime。"""
from .shell import ShellTool
from .files import FilesTool
from .http import HttpTool
from .search import SearchTool
from .calc import CalcTool
from .browser import BrowserTool
from .time import TimeTool
from .web import WebTool
from ..agent.tools import ToolRegistry


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for cls in (ShellTool, FilesTool, HttpTool, SearchTool, CalcTool, BrowserTool, TimeTool, WebTool):
        t = cls()
        reg.register_func(t.name, t.description, t.run, t.parameters)
    return reg


__all__ = ["build_default_registry", "ShellTool", "FilesTool", "HttpTool",
           "SearchTool", "CalcTool", "BrowserTool", "TimeTool", "WebTool"]
