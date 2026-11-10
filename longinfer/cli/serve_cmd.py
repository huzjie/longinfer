# -*- coding: utf-8 -*-
"""启动 FastAPI 控制面。"""
from __future__ import annotations

from ..logging import get_logger

log = get_logger("longinfer.serve")


def serve(config, args) -> int:
    import uvicorn
    from ..api.app import create_app
    api = config["api"]
    host = args.host or api.get("host", "127.0.0.1")
    port = args.port or api.get("port", 8000)
    app = create_app(config)
    log.info("控制面启动: http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
    return 0
