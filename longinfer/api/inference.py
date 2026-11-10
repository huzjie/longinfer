# -*- coding: utf-8 -*-
"""推理服务层：把 provider 与引擎组合，向上层提供统一接口。"""
from __future__ import annotations

from ..config import Config
from ..logging import get_logger

log = get_logger("longinfer.inference")


def build_provider(config: Config):
    """按配置创建 provider 实例。"""
    name = config["provider"]
    model = config["model"]
    base_url = config.get("base_url")
    api_key = config.get("api_key")
    if name == "mock":
        from ..providers.mock import MockProvider
        return MockProvider(model)
    import importlib
    from ..providers.base import Provider
    from ..exceptions import ProviderError
    mod = importlib.import_module(f"longinfer.providers.{name}")
    cls = None
    for attr in vars(mod).values():
        if isinstance(attr, type) and issubclass(attr, Provider) and attr is not Provider \
                and attr.__module__ == mod.__name__:
            cls = attr
            break
    if cls is None:
        raise ProviderError(f"provider 类不存在: {name}")
    return cls(model, base_url, api_key)


class InferenceService:
    def __init__(self, config: Config):
        self.config = config
        self.provider = build_provider(config)

    def chat(self, prompt: str, max_new: int = 256, temperature: float = 0.7) -> dict:
        text = self.provider.complete(prompt, max_new, temperature)
        return {"text": text, "provider": self.config["provider"], "model": self.config["model"],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(text)}}
