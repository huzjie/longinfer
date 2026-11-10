# Kubernetes / Helm

## Helm

```bash
helm install longinfer ./deploy/helm
```

## 原生 K8s

```bash
kubectl apply -f deploy/k8s/
```

默认含：deployment、service、hpa、configmap、ingress。国产算力场景请调整 `EP`/`VP` 分片参数。
