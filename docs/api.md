# REST API

启动后访问 `http://127.0.0.1:8000/docs` 查看交互式文档。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | /health | 健康检查 |
| GET | /v1/cluster/plan | 分布式部署计划 |
| POST | /v1/chat | 对话推理 |
| POST | /v1/vision/tokenize | 视觉 RVQ 离散化 |
| POST | /v1/agent/run | 长程智能体 |
| POST | /v1/quant/demo | 量化演示 |
| GET | /console | Web 控制台 |
