# -*- coding: utf-8 -*-
"""`python -m longinfer.api` 启动控制面。"""
from __future__ import annotations

import uvicorn

from ..config import Config
from .app import create_app


def main():
    config = Config.discover()
    api = config["api"]
    app = create_app(config)
    uvicorn.run(app, host=api.get("host", "127.0.0.1"), port=api.get("port", 8000))


if __name__ == "__main__":
    main()
