# -*- coding: utf-8 -*-
"""在本地知识切片里做关键词检索"""
from __future__ import annotations


class SearchTool:
    name = "search"
    description = "在本地知识切片里做关键词检索"
    parameters = {"query": "str"}

    def run(self, **kwargs):
        q = kwargs.get("query", "").lower()
        return f"[search] 离线检索「{q}」未接入外部索引，返回占位结果（可替换为向量检索）"

