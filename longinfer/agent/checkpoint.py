# -*- coding: utf-8 -*-
"""检查点：把轨迹/状态序列化到磁盘，支持断点续跑。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .trajectory import Trajectory
from ..exceptions import CheckpointError
from ..utils.hash import content_hash


def save_checkpoint(traj: Trajectory, path: str) -> str:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = traj.to_dict()
    payload["_hash"] = content_hash(payload)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(dest)


def load_checkpoint(path: str) -> Trajectory:
    p = Path(path)
    if not p.exists():
        raise CheckpointError(f"检查点不存在: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    traj = Trajectory(goal=data.get("goal", ""))
    from .trajectory import Step
    for s in data.get("steps", []):
        traj.add(Step(**s))
    return traj


def list_checkpoints(dir_path: str) -> list:
    d = Path(dir_path)
    if not d.exists():
        return []
    return sorted(str(p) for p in d.glob("*.json"))
