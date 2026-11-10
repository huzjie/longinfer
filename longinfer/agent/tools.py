# -*- coding: utf-8 -*-
"""工具注册中心：长程智能体可调用的工具集合。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: dict = None

    def run(self, **kwargs) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_func(self, name: str, description: str, func: Callable, parameters: Optional[dict] = None) -> None:
        self.register(Tool(name, description, func, parameters))

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> list:
        return sorted(self._tools.keys())

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())
