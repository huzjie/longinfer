# -*- coding: utf-8 -*-
"""`longinfer` 命令行入口（argparse，零第三方依赖）。"""
from __future__ import annotations

import argparse
import sys

from ..version import __version__
from ..logging import get_logger

log = get_logger("longinfer")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="longinfer",
        description="长程多模态 MoE 智能体推理与部署引擎",
    )
    p.add_argument("--version", action="version", version=f"longinfer {__version__}")
    p.add_argument("--config", default=None, help="配置文件路径（默认自动发现 config.yaml）")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="自检：逐项诊断六个子系统")

    r = sub.add_parser("run", help="离线完整跑一遍推理链路")
    r.add_argument("prompt", nargs="?", default="解释 MoE 稀疏路由")
    r.add_argument("--provider", default=None)

    c = sub.add_parser("chat", help="交互式对话")
    c.add_argument("--provider", default=None)

    v = sub.add_parser("vision-tokenize", help="图像 RVQ 离散化")
    v.add_argument("image", nargs="?", default="")
    v.add_argument("--levels", type=int, default=4)
    v.add_argument("--patch-size", type=int, default=16)

    q = sub.add_parser("quant", help="W4A4 量化演示/执行")
    q.add_argument("--method", default="w4a4", choices=["w4a4", "w8a8", "fp8", "gptq", "awq"])
    q.add_argument("--rows", type=int, default=16)
    q.add_argument("--cols", type=int, default=64)

    a = sub.add_parser("agent-run", help="运行长程智能体")
    a.add_argument("--goal", default="修复 checkout 页崩溃")
    a.add_argument("--max-steps", type=int, default=20)
    a.add_argument("--resume", default=None)

    s = sub.add_parser("serve", help="启动 FastAPI 控制面")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)

    b = sub.add_parser("benchmark", help="基准：MoE 路由/量化/采样吞吐")
    b.add_argument("--tokens", type=int, default=10000)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from ..config import Config
    config = Config.load(args.config)

    if args.command == "doctor":
        from .doctor_cmd import run_doctor
        return run_doctor(config)
    if args.command == "run":
        from .run_cmd import run
        return run(config, args)
    if args.command == "chat":
        from .chat_cmd import chat
        return chat(config, args)
    if args.command == "vision-tokenize":
        from .vision_cmd import vision
        return vision(config, args)
    if args.command == "quant":
        from .quant_cmd import quant
        return quant(config, args)
    if args.command == "agent-run":
        from .agent_cmd import agent_run
        return agent_run(config, args)
    if args.command == "serve":
        from .serve_cmd import serve
        return serve(config, args)
    if args.command == "benchmark":
        from .benchmark_cmd import benchmark
        return benchmark(config, args)
    log.error("未知命令")
    return 1


if __name__ == "__main__":
    sys.exit(main())
