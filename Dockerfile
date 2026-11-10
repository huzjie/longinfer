# ---- 多阶段构建：builder 产 wheel，runtime 精简 ----
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY longinfer ./longinfer
RUN pip wheel --no-cache-dir --wheel-dir /wheels -e ".[api,yaml]"

FROM python:3.11-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /wheels
COPY --from=builder /app/longinfer ./longinfer
COPY pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir --no-index --find-links=/wheels longinfer
COPY config.example.yaml ./config.yaml
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://127.0.0.1:8000/health || exit 1
ENV LFI_PROVIDER=mock
ENTRYPOINT ["longinfer"]
CMD ["serve", "--port", "8000"]
