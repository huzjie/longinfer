# -*- coding: utf-8 -*-
"""SDK 类型定义。"""
from __future__ import annotations

from typing import Any, List, Optional, TypedDict


class ChatResult(TypedDict):
    text: str
    provider: str
    model: str
    usage: dict
