# -*- coding: utf-8 -*-
"""量化服务。"""
from __future__ import annotations

from fastapi import APIRouter
from . import schemas

router = APIRouter(tags=["quant"])


@router.post("/v1/quant/demo")
def demo(req: schemas.QuantRequest):
    from ...quant.int4 import QuantizedMatrix
    from ...utils.tensor import randn
    W = randn((8, 32), seed=0)
    qm = QuantizedMatrix(8, 32, req.group_size, req.sym).quantize(W)
    dq = qm.dequantize()
    return {"method": req.method, "rows": 8, "cols": 32, "nbytes": qm.nbytes(),
            "max_abs_error": max(abs(a - b) for ra, rb in zip(W, dq) for a, b in zip(ra, rb))}
