# -*- coding: utf-8 -*-
"""视觉 RVQ 离散化。"""
from __future__ import annotations

from fastapi import APIRouter
from . import schemas

router = APIRouter(tags=["vision"])


@router.post("/v1/vision/tokenize", response_model=schemas.VisionResponse)
def tokenize(req: schemas.VisionRequest):
    from ...vision.image_loader import load_image
    from ...vision.tokenizer import VisionTokenizer
    img = load_image(req.image_path or "")
    tok = VisionTokenizer(patch_size=req.patch_size, rvq_levels=req.rvq_levels)
    tokens = tok.encode(img)
    return {"num_patches": len(tokens), "tokens": tokens[:256]}
