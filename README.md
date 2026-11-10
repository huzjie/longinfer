# longinfer · 长程多模态 MoE 智能体推理与部署引擎

> 面向万亿参数级「原生多模态 + 长程智能体」大模型的一站式**推理、量化、编排与分布式部署**平台。
> 以 2026-11-09 美团开源 **LongCat-2.5**（1.6T 总参 / 48B 激活 MoE / 100 万上下文 / 视觉原生 token 化 / MIT 协议）为技术蓝本，
> 把「稀疏 MoE 路由、W4A4 量化、视觉残差向量量化、长程智能体运行时、国产算力分布式部署」这套方法论，工程化成一个可部署、可复现、可审计的开源平台。

---

## 这是什么

longinfer 不是一个「模型」，而是一套**围绕长程多模态 MoE 大模型的服务化运行时**。它把 LongCat-2.5 这类模型背后的关键技术拆成六个可独立使用的子系统，并全部用**纯 Python 标准库**实现为可离线运行、可验证的参考内核：

| 子系统 | 解决的问题 | 对应 LongCat-2.5 技术点 |
|---|---|---|
| `moe` | 万亿参数模型怎么只激活 48B 就完成一次推理 | MoE 稀疏路由 + Top-K 专家选择 + 辅助负载均衡损失 |
| `quant` | 1.8TB 权重怎么塞进受限显存 | W4A4（权重/激活同 4bit）+ INT4 打包 + GPTQ/AWQ 量化与反量化 |
| `vision` | 图像如何与文本共享同一套参数 | 多层级离散 token + 残差向量量化（RVQ）+ 视觉码本联合训练 |
| `agent` | 长任务怎么中断续跑、怎么避免过度思考 | partial rollout 异步训练 + 检查点断点续跑 + LLM 轨迹评判 |
| `dist` | 国产算力显存/带宽受限怎么横向扩展 | 张量/专家并行分片 + 动态专家复制 + 混合词表并行 + 零拷贝通信 |
| `serve` | 怎么对外提供生产级推理服务 | FastAPI 控制面 + 连续批处理调度 + 多 Provider 适配 + Web 控制台 |

每个子系统都带**离线可跑**的参考实现（`mock` / 纯标准库内核），接上真实后端（vLLM / SGLang / Ollama / OpenAI 兼容 / 长猫官方 API）即可直接对外服务。

---

## 为什么值得看（技术原理速览）

**1. MoE 为什么能「1.6T 参数、48B 激活」？**
MoE 把 FFN 层替换成 N 个「专家」，每个 token 只被路由器（router）分给 Top-K 个专家计算。计算量只跟「激活参数」挂钩，模型「知识容量」却跟「总参数」挂钩。代价是：路由器可能「偏心」，导致少数专家过载、其余闲置。解法是**辅助负载均衡损失**（aux-loss）+ **容量因子**（capacity factor）做专家配额的显式约束——`moe/` 子系统完整实现了这套路由 + 均衡逻辑。

**2. W4A4 量化为什么敢说「精度基本不降」？**
权重和激活同时压到 4bit（W4A4），配合原生 INT4 矩阵乘单元。关键在**逐通道/逐组缩放因子**（per-channel scale）与**激活的动态校准**（calibration）——`quant/` 实现了 GPTQ（逐列最小二乘误差补偿）与 AWQ（按激活幅值保护重要通道）两种量化范式，以及 INT4 的位打包/解包内核。

**3. 视觉怎么「原生」进模型？**
不是外挂一个 ViT 再拼特征，而是把图像按 patch 切分后，对特征做**残差向量量化（RVQ）**，逐级补足细节，生成多层级离散 token，与文本 token 在同一词表、同一套参数里协同训练。`vision/` 完整实现了 patchify → 特征编码 → RVQ 逐级量化 → 码本联合生成的链路。

**4. 长程智能体怎么「中断续跑、不钻空子」？**
真实长任务的训练/执行会被中断。partial rollout 把没跑完的轨迹**序列化到检查点**，恢复后接着跑；用 LLM 当「评判员」对重复推理、冗余工具调用、过度思辨施加显式约束。`agent/` 实现了带检查点、可重放的 rollout 运行时 + 轨迹评判器 + 工具注册中心。

---

## 快速开始

```bash
git clone https://github.com/huzjie/longinfer.git
cd longinfer
pip install -e .            # 零第三方依赖即可跑内核；[api] 安装 FastAPI 控制面

longinfer doctor             # 自检：六个子系统逐项诊断
longinfer run "计算 (2+3)*4 并解释 MoE 路由" --provider mock   # 离线完整跑一遍
longinfer quant --model ./ckpt --w4a4 --calibrate samples.jsonl   # W4A4 量化
longinfer vision-tokenize ./demo.jpg --levels 4                    # 视觉 RVQ 离散化
longinfer agent-run --goal "修复 checkout 页崩溃" --checkpoint ./.lfi   # 长程智能体
longinfer serve --port 8000                                       # REST API + Web 控制台
```

接真实后端（示例接 vLLM 部署 LongCat-2.5 量化版）：

```bash
export LFI_PROVIDER=vllm LFI_MODEL=LongCat-2.5 LFI_BASE_URL=http://127.0.0.1:8000/v1
longinfer serve --port 8001
```

---

## 仓库结构（200+ 文件）

```
longinfer/
├── longinfer/            # 核心包（纯标准库内核 + 可选 FastAPI 控制面）
│   ├── moe/              # 稀疏 MoE 路由 / 专家 / 负载均衡
│   ├── quant/            # W4A4 · INT4 打包 · GPTQ · AWQ · FP8
│   ├── vision/           # patchify · RVQ · 码本 · 图像离散 token 化
│   ├── agent/            # 长程 rollout · 检查点 · 轨迹 · 评判器 · 工具
│   ├── dist/             # 张量/专家/词表并行 · 分片计划 · 零拷贝通信
│   ├── engine/           # 采样 · KV 缓存 · 连续批处理调度 · 模型加载
│   ├── providers/        # mock/vllm/sglang/ollama/openai/glm/qwen/kimi/deepseek/longcat
│   ├── tools/            # shell/files/http/search/code/calc/browser/time/web
│   ├── api/              # FastAPI 控制面 + 路由 + 中间件
│   ├── cli/              # doctor/run/chat/vision/quant/agent-run/serve/benchmark
│   └── sdk/              # Python / TypeScript 客户端
├── deploy/               # k8s · Helm（含国产算力分片与专家并行配置）
├── docs/                 # 25+ 篇技术文档
├── examples/             # 可运行示例
├── tests/                # 单测（纯标准库 unittest）
├── .github/workflows/    # CI / Release / CodeQL
├── Dockerfile            # 多阶段构建
└── config.example.yaml   # 全量配置模板
```

---

## 许可证

MIT License。参见 [LICENSE](LICENSE)。

> 说明：本项目为**独立开源实现**，与美团 / LongCat 团队无隶属关系；`longcat` provider 仅为对接其公开 API / 权重部署的适配器。
