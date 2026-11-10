# -*- coding: utf-8 -*-
"""同步便捷函数。"""
from __future__ import annotations

from .client import LonginferClient


def chat(base_url: str, prompt: str, **kw) -> dict:
    return LonginferClient(base_url).chat(prompt, **kw)
