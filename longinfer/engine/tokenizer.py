# -*- coding: utf-8 -*-
"""文本分词器：内置朴素分词（离线可跑）；真实模型走 provider 自带 tokenizer。"""
from __future__ import annotations

import re
from typing import List


class TextTokenizer:
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError


class SimpleTokenizer(TextTokenizer):
    """字符级分词：把 UTF-8 字节映射为 token id，保证任意文本离线可编码/解码。"""

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        return bytes(b for b in ids).decode("utf-8", errors="replace")

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))
