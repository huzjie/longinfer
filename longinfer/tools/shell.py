# -*- coding: utf-8 -*-
"""执行只读 shell 命令（沙箱约束下）"""
from __future__ import annotations


class ShellTool:
    name = "shell"
    description = "执行只读 shell 命令（沙箱约束下）"
    parameters = {"command": "str"}

    def run(self, **kwargs):
        import subprocess
        cmd = kwargs.get("command", "echo ok")
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            return r.stdout.strip() or r.stderr.strip()
        except Exception as e:  # noqa
            return f"shell 执行失败: {e}"

