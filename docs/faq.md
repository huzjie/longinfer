# 常见问题

**Q: 需要 GPU 吗？**
A: 内核离线可跑，不需要 GPU；跑真实大模型需接 vLLM/SGLang 后端。

**Q: 支持 LongCat-2.5 权重吗？**
A: 通过 `longcat` provider 对接官方 API，或用 vLLM/SGLang 部署官方权重后对接。

**Q: 为什么内核是纯标准库？**
A: 保证离线可复现、可审计，方法论清晰可学。
