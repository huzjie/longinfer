# -*- coding: utf-8 -*-
"""原生多模态：把图像编码为多层级离散 token，与文本共享同一语义空间。

对应 LongCat-2.5 的「视觉信号统一编码为多层级离散词元 + 残差向量量化」。
"""
from .patchify import patchify
from .codebook import Codebook, init_codebook
from .rvq import ResidualVQ
from .tokenizer import VisionTokenizer, image_to_tokens
from .image_loader import load_image

__all__ = ["patchify", "Codebook", "init_codebook", "ResidualVQ", "VisionTokenizer", "image_to_tokens", "load_image"]
