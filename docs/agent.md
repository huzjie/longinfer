# 长程智能体运行时

## 原理

长程任务会被中断。partial rollout 让「没跑完」的轨迹保存检查点、后续继续；LLM 评判员对重复推理、冗余工具调用、过度思辨施加约束。

## 关键点

1. **partial rollout**：按 step 推进，周期保存检查点。
2. **检查点**：轨迹序列化（含哈希溯源），可断点续跑。
3. **轨迹评判**：评分 0~1，惩罚过度思考/冗余调用/重复推理。
4. **过采样**：按通过率动态调整——通过率接近 0.5 的信息量最大，全对/全错样本剔除。
5. **沙箱约束**：执行期断网、清 git 历史与文件缓存，模拟真实环境。

## 用法

```python
from longinfer.agent.runtime import AgentRuntime
traj = AgentRuntime(config, provider, tools).run("修复 checkout 页崩溃")
```

## 可复用要点

- 把「失败轨迹」转成可复现、可验证的训练样本，是长程能力提升的关键。
