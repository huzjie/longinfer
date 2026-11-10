# 快速开始

## 1. 安装

```bash
git clone https://github.com/huzjie/longinfer.git
cd longinfer
pip install -e .              # 零第三方依赖，内核离线可跑
pip install -e ".[api,yaml]"  # 安装 FastAPI 控制面 + YAML 支持
```

## 2. 自检

```bash
longinfer doctor
```

会逐项诊断六个子系统（MoE 路由 / 量化 / 视觉 / 智能体 / 分布式 / 引擎）与 provider 调用。

## 3. 离线跑一遍

```bash
longinfer run "解释 MoE 稀疏路由的原理" --provider mock
```

## 4. 量化演示

```bash
longinfer quant --method w4a4 --rows 16 --cols 64
```

## 5. 视觉离散化

```bash
longinfer vision-tokenize ./demo.jpg --levels 4
```

## 6. 长程智能体

```bash
longinfer agent-run --goal "修复 checkout 页崩溃" --max-steps 20
```

## 7. 启动控制面

```bash
longinfer serve --port 8000
# 打开 http://127.0.0.1:8000/docs 或 /console
```

## 8. 接真实后端

```bash
export LFI_PROVIDER=vllm LFI_MODEL=LongCat-2.5 LFI_BASE_URL=http://127.0.0.1:8000/v1
longinfer serve --port 8001
```
