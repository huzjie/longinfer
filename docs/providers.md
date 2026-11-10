# Provider 后端

| Provider | 说明 |
|---|---|
| mock | 离线确定性，零依赖 |
| vllm | vLLM（OpenAI 兼容） |
| sglang | SGLang（OpenAI 兼容） |
| ollama | 本地 Ollama |
| openai | 任意 OpenAI 兼容后端 |
| glm / qwen / kimi / deepseek | 国产模型官方 API |
| longcat | LongCat 官方 API 或自建部署 |

切换后端：设置 `LFI_PROVIDER` / `LFI_MODEL` / `LFI_BASE_URL` / `LFI_API_KEY`。
