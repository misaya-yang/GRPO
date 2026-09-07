# 采样与数值修复结果

整数累计概率替换了有偏的尾部修补；FP32 base、CPU embedding、激活卸载使实际 Qwen7B 的生成/score 校验通过。模型、K/N、两臂估计目标未变。

随后用 NumPy 整数 limb 前缀和加速 CDF。8 条原 Dev 回答精确重放通过：tokens、logp、RNG、reward、release 均一致，宏组成本从约 154.8 秒降至 77.0 秒，新增研究样本为零。见 [fast_replay_receipt.json](fast_replay_receipt.json)。这是同输出实现加速，不是 Strat-full 对 IID-all 的收益。

`numeric_receipt.json` 是实际模型数值验证；`change.json` 保留向量化之前的源码 hash 与当时状态快照，不代表当前源码版本。后续 Dev 和 256 回答小规模实测均完成，见[最终报告](../trend_pilot/REPORT.md)。Luna 不再接管运行。

提交前本机完整检查：139 tests passed，lint 与 exact 通过；数值通过不构成方法收益或在线训练结论。
