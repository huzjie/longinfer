# -*- coding: utf-8 -*-
"""设备拓扑：描述超节点内的多卡资源。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Device:
    device_id: int
    memory_gb: float
    bandwidth_gbps: float = 900.0
    name: str = "gpu"


@dataclass
class Topology:
    devices: List[Device] = field(default_factory=list)
    node_name: str = "supernode-0"

    @classmethod
    def from_memory(cls, world_size: int, per_card_gb: float = 80.0) -> "Topology":
        return cls([Device(i, per_card_gb) for i in range(world_size)])

    def total_memory_gb(self) -> float:
        return sum(d.memory_gb for d in self.devices)

    def num_devices(self) -> int:
        return len(self.devices)
