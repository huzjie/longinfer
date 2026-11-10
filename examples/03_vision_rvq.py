# -*- coding: utf-8 -*-
"""示例 3：视觉 RVQ 离散 token 化。"""
from longinfer.vision.image_loader import load_image
from longinfer.vision.tokenizer import VisionTokenizer

img = load_image("")  # 合成渐变图
tok = VisionTokenizer(patch_size=16, rvq_levels=4)
tokens = tok.encode(img)
print("patch 数:", len(tokens))
print("每个 patch 的 4 级码下标示例:", tokens[0])
