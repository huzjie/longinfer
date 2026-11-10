# -*- coding: utf-8 -*-
"""月之暗面 Kimi 后端（OpenAI 兼容接口）。"""
from __future__ import annotations

from .base import Provider


class KimiProvider(Provider):
    name = "Kimi".lower()
    openai_compatible = True

    def complete(self, prompt: str, max_new: int = 256, temperature: float = 0.7) -> str:
        url = (self.base_url or "").rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new,
            "temperature": temperature,
        }
        data = self._post_json(url, payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            from ..exceptions import ProviderError
            raise ProviderError(f"{self.name} 响应格式异常: {data}") from e
