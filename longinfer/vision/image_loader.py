# -*- coding: utf-8 -*-
"""图像加载：PGM（P2/P5）原生支持；PNG/JPEG 可选 PIL；否则合成测试图。"""
from __future__ import annotations

from typing import List


def _synthetic(size: int = 64) -> List[List[float]]:
    """生成确定性渐变测试图（离线可复现）。"""
    img = []
    for y in range(size):
        row = []
        for x in range(size):
            row.append(float((x + y) % 256) / 255.0)
        img.append(row)
    return img


def load_image(path: str, size: int = 64) -> List[List[float]]:
    """加载图像为 [0,1] 灰度矩阵；无法识别时回退到合成图。"""
    try:
        with open(path, "rb") as f:
            magic = f.read(2)
        if magic == b"P5":
            return _load_pgm_p5(path)
        if magic == b"P2":
            return _load_pgm_p2(path)
        # 尝试 PIL
        try:
            from PIL import Image  # type: ignore
            im = Image.open(path).convert("L")
            im = im.resize((size, size))
            px = list(im.getdata())
            return [px[y * size:(y + 1) * size] for y in range(size)]
        except Exception:
            return _synthetic(size)
    except Exception:
        return _synthetic(size)


def _load_pgm_p5(path: str) -> List[List[float]]:
    with open(path, "rb") as f:
        data = f.read()
    parts = data.split(b"\n", 3)
    hdr = parts[1].split()
    w, h = int(hdr[0]), int(hdr[1])
    raw = parts[3]
    img = []
    for y in range(h):
        row = [float(raw[y * w + x]) / 255.0 for x in range(w)]
        img.append(row)
    return img


def _load_pgm_p2(path: str) -> List[List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        tokens = f.read().split()
    # P2 W H maxval 之后是像素
    idx = 4
    w, h = int(tokens[1]), int(tokens[2])
    img = []
    for y in range(h):
        row = [float(tokens[idx + y * w + x]) / 255.0 for x in range(w)]
        img.append(row)
    return img
