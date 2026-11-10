# Docker 部署

```bash
docker build -t longinfer:latest .
docker run --rm -p 8000:8000 -e LFI_PROVIDER=mock longinfer:latest
# 或
docker compose up --build
```

多阶段构建：builder 产 wheel，runtime 精简镜像。
