# -*- coding: utf-8 -*-
"""Mock provider：确定性离线应答，零网络依赖。"""
from __future__ import annotations

from .base import Provider
from ..engine.mock_model import MockModel
from ..config import Config


class MockProvider(Provider):
    name = "mock"

    def __init__(self, model: str = "longcat-2.5-mock"):
        super().__init__(model)
        self._model = MockModel(Config())

    def complete(self, prompt: str, max_new: int = 256, temperature: float = 0.7) -> str:
        return self._model.complete(prompt, max_new)
