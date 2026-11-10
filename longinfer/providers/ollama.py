# -*- coding: utf-8 -*-
"""Ollama 本地推理后端。"""
from __future__ import annotations

from .base import Provider
from ..exceptions import ProviderError


class OllamaProvider(Provider):
    name = "ollama"
    openai_compatible = False

    def complete(self, prompt: str, max_new: int = 256, temperature: float = 0.7) -> str:
        url = (self.base_url or "http://127.0.0.1:11434").rstrip("/") + "/api/generate"
        payload = {"model": self.model, "prompt": prompt,
                   "stream": False, "options": {"temperature": temperature, "num_predict": max_new}}
        data = self._post_json(url, payload)
        if "response" not in data:
            raise ProviderError(f"Ollama 响应异常: {data}")
        return data["response"]
