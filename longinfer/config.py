# -*- coding: utf-8 -*-
"""配置加载：PyYAML 优先，失败回退到内置 mini-YAML；环境变量 LFI_* 可覆盖任意键。"""
from __future__ import annotations

import os
from pathlib import Path

from ._yaml import load as _mini_load
from .exceptions import ConfigError

DEFAULTS = {
    "home": "~/.longinfer",
    "models_dir": "~/.longinfer/models",
    "checkpoint_dir": "~/.longinfer/checkpoints",
    "log_dir": "~/.longinfer/logs",
    "log_level": "INFO",
    "provider": "mock",
    "model": "longcat-2.5-mock",
    "base_url": None,
    "api_key": None,
    "moe": {
        "num_experts": 256,
        "top_k": 8,
        "capacity_factor": 1.25,
        "aux_loss_coef": 0.01,
        "router_jitter": 0.0,
        "expert_parallel_size": 8,
    },
    "quant": {
        "method": "w4a4",
        "group_size": 128,
        "sym": True,
        "protect_outliers": True,
        "calibrate_samples": 128,
    },
    "vision": {
        "patch_size": 16,
        "image_size": 1024,
        "rvq_levels": 4,
        "codebook_size": 16384,
        "code_dim": 256,
    },
    "agent": {
        "max_steps": 200,
        "checkpoint_every": 10,
        "partial_rollout": True,
        "judge_provider": "mock",
        "oversample_by_pass_rate": True,
        "sandbox": {"offline": True, "clear_git_history": True, "clear_file_cache": True},
    },
    "dist": {
        "world_size": 1,
        "tensor_parallel": 1,
        "expert_parallel": 8,
        "vocab_parallel": 2,
        "dynamic_expert_replication": True,
        "zero_copy_comm": True,
    },
    "api": {"host": "127.0.0.1", "port": 8000, "auth_token": None, "max_batch_size": 256, "max_context": 1000000},
    "observability": {"tracing": True, "metrics": True, "replay": True, "audit": True},
}

_ENV_PREFIX = "LFI_"


def _try_pyyaml(path: str):
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_env(data: dict, prefix: str = _ENV_PREFIX) -> dict:
    out = dict(data)
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        sub = key[len(prefix):].lower()
        if "_" not in sub:
            out[sub] = _coerce(val)
        else:
            section, field = sub.split("_", 1)
            if section in out and isinstance(out[section], dict):
                out[section][field] = _coerce(val)
    return out


def _coerce(val: str):
    low = val.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", ""):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


class Config:
    def __init__(self, data: dict | None = None):
        self._data = _deep_merge(DEFAULTS, data or {})

    def __getitem__(self, key: str):
        return self._data[key]

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def as_dict(self) -> dict:
        import copy
        return copy.deepcopy(self._data)

    def resolve(self, key: str, default: str | None = None) -> str | None:
        val = self._data.get(key, default)
        return _expand(val) if isinstance(val, str) else val

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        data: dict = {}
        if path and Path(path).exists():
            data = _try_pyyaml(path)
            if data is None:
                data = _mini_load(path)
            if not isinstance(data, dict):
                raise ConfigError(f"配置根节点必须是映射: {path}")
        data = _apply_env(data)
        return cls(data)

    @classmethod
    def discover(cls) -> "Config":
        for cand in ("config.yaml", "config.yml", "config.example.yaml"):
            if Path(cand).exists():
                return cls.load(cand)
        return cls.load(None)
