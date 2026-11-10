# -*- coding: utf-8 -*-
"""图像 RVQ 离散化。"""
from __future__ import annotations

from ..logging import get_logger

log = get_logger("longinfer.vision")


def vision(config, args) -> int:
    from ..vision.image_loader import load_image
    from ..vision.tokenizer import VisionTokenizer
    img = load_image(args.image or "")
    tok = VisionTokenizer(patch_size=args.patch_size, rvq_levels=args.levels)
    tokens = tok.encode(img)
    log.info("图像 %s -> %d patches，每个 %d 级离散 token", args.image or "(合成图)", len(tokens), args.levels)
    print("tokens[:8] =", tokens[:8])
    return 0
