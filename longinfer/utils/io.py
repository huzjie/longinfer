# -*- coding: utf-8 -*-
"""文件读写辅助：JSONL、原子写、路径展开。"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write_text(path: str, text: str) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, str(dest))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def expand(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))
