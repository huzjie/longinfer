# -*- coding: utf-8 -*-
"""LLM 后端适配器：统一 complete/chat 接口，provider 可插拔。"""
from .base import Provider, ProviderFactory
from .mock import MockProvider
from .openai import OpenAIProvider
from .ollama import OllamaProvider
from .vllm import VLLMProvider
from .sglang import SGLangProvider
from .glm import GLMProvider
from .qwen import QwenProvider
from .kimi import KimiProvider
from .deepseek import DeepSeekProvider
from .longcat import LongCatProvider

__all__ = [
    "Provider", "ProviderFactory", "MockProvider", "OpenAIProvider",
    "OllamaProvider", "VLLMProvider", "SGLangProvider", "GLMProvider",
    "QwenProvider", "KimiProvider", "DeepSeekProvider", "LongCatProvider",
]
