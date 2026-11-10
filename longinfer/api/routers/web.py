# -*- coding: utf-8 -*-
"""静态控制台路由：把单文件 Web 控制台挂到 /console。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["web"])

_CONSOLE = Path(__file__).resolve().parent.parent.parent / "web" / "console.html"


@router.get("/console", response_class=HTMLResponse)
def console():
    return _CONSOLE.read_text(encoding="utf-8") if _CONSOLE.exists() else "<h1>console.html 缺失</h1>"
