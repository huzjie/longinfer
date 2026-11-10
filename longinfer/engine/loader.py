# -*- coding: utf-8 -*-
"""模型加载：按 provider 分发到真实后端，或构建离线 mock 模型。"""
from __future__ import annotations

from ..config import Config
from ..logging import get_logger

log = get_logger("longinfer.loader")


class ModelLoader:
    def __init__(self, config: Config):
        self.config = config

    def load(self):
        provider = self.config["provider"]
        model = self.config["model"]
        if provider == "mock":
            from .mock_model import MockModel
            return MockModel(self.config)
        log.info("加载模型 %s (provider=%s)", model, provider)
        # 真实后端由 providers 适配器负责，这里只返回配置引用
        return {"provider": provider, "model": model}
