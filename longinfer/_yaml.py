# -*- coding: utf-8 -*-
"""极简 YAML 子集解析器（无第三方依赖时的兜底）。

支持：标量(字符串/整数/浮点/布尔/null)、嵌套映射、序列、注释。
不支持的复杂特性（锚点、多行块、流式集合）会抛 ConfigError，提示安装 PyYAML。
"""
from __future__ import annotations

from .exceptions import ConfigError


class _Line:
    __slots__ = ("raw", "indent", "key", "value")

    def __init__(self, raw: str, indent: int, key: str, value: str):
        self.raw = raw
        self.indent = indent
        self.key = key
        self.value = value


def _parse_scalar(text: str):
    text = text.strip()
    if not text or text == "~":
        return None
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none"):
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    if text.startswith('"') or text.startswith("'"):
        return text.strip("\"'")
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _strip_comment(line: str) -> str:
    in_s = False
    in_d = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            return line[:i]
    return line


def _parse_inline_seq(value: str):
    """解析 [a, b, c] 形式的内联序列。"""
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return None
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [_parse_scalar(part) for part in inner.split(",")]


def _split_key_value(line: str):
    line = _strip_comment(line)
    for i, ch in enumerate(line):
        if ch == ":":
            return line[:i].strip(), line[i + 1:].strip()
    return line.strip(), ""


def load(text: str):
    lines = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if raw.lstrip().startswith("#"):
            continue
        key, value = _split_key_value(raw)
        lines.append(_Line(raw, indent, key, value))

    def build(start: int, indent: int):
        result: dict = {}
        i = start
        while i < len(lines):
            ln = lines[i]
            if ln.indent < indent:
                break
            if ln.indent > indent:
                raise ConfigError(f"意外的缩进: {ln.raw!r}")
            if ln.value == "" or ln.value.startswith(("- ", "[", "{")):
                # 嵌套映射或序列
                if ln.value.startswith(("- ",)):
                    seq = [ln.value[1:].strip()]
                    j = i + 1
                    while j < len(lines) and lines[j].indent > indent and lines[j].value.startswith("- "):
                        seq.append(lines[j].value[1:].strip())
                        j += 1
                    result[ln.key] = [_parse_scalar(s) for s in seq]
                    i = j
                    continue
                inline = _parse_inline_seq(ln.value)
                if inline is not None:
                    result[ln.key] = inline
                    i += 1
                    continue
                if i + 1 < len(lines) and lines[i + 1].indent > indent:
                    result[ln.key], i = build(i + 1, lines[i + 1].indent)
                    continue
                result[ln.key] = None
                i += 1
            else:
                result[ln.key] = _parse_scalar(ln.value)
                i += 1
        return result, i

    data, _ = build(0, 0)
    return data


def load_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return load(f.read())
