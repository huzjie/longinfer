# -*- coding: utf-8 -*-
"""Longinfer Python 客户端（urllib，零依赖）。"""
from __future__ import annotations

import json
import urllib.request
from typing import Optional


class LonginferClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", token: Optional[str] = None, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(self.base_url + path)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def health(self) -> dict:
        return self._get("/health")

    def chat(self, prompt: str, max_new: int = 256, temperature: float = 0.7) -> dict:
        return self._post("/v1/chat", {"prompt": prompt, "max_new": max_new, "temperature": temperature})

    def vision_tokenize(self, image_path: str = "", patch_size: int = 16, rvq_levels: int = 4) -> dict:
        return self._post("/v1/vision/tokenize", {"image_path": image_path, "patch_size": patch_size, "rvq_levels": rvq_levels})

    def agent_run(self, goal: str, max_steps: int = 200) -> dict:
        return self._post("/v1/agent/run", {"goal": goal, "max_steps": max_steps})

    def cluster_plan(self) -> dict:
        return self._get("/v1/cluster/plan")
