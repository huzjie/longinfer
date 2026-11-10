# -*- coding: utf-8 -*-
"""Provider 基类与工厂。"""
from __future__ import annotations

import json
import urllib.request
from typing import Optional

from ..exceptions import ProviderError
from ..logging import get_logger

log = get_logger("longinfer.provider")


class Provider:
    """统一推理后端接口。"""

    name = "base"
    openai_compatible = True

    def __init__(self, model: str = "", base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def complete(self, prompt: str, max_new: int = 256, temperature: float = 0.7) -> str:
        raise NotImplementedError

    def chat(self, messages: list, max_new: int = 256, temperature: float = 0.7) -> str:
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return self.complete(prompt, max_new, temperature)

    def _post_json(self, url: str, payload: dict, timeout: int = 120) -> dict:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa
            raise ProviderError(f"调用 {self.name} 失败: {e}") from e


class ProviderFactory:
    """按名称创建 provider。"""

    REGISTRY: dict = {}

    @classmethod
    def register(cls, name: str, builder):
        cls.REGISTRY[name] = builder

    @classmethod
    def create(cls, name: str, model: str = "", base_url: Optional[str] = None, api_key: Optional[str] = None) -> Provider:
        if name == "mock":
            from .mock import MockProvider
            return MockProvider(model)
        builder = cls.REGISTRY.get(name)
        if builder is None:
            raise ProviderError(f"未知 provider: {name}，可选: {sorted(cls.REGISTRY) + ['mock']}")
        return builder(model, base_url, api_key)
