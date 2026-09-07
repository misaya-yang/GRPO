# 当前状态：v5.1 小规模实测完成，离线诊断

2026-09-07：`v5_1_20260907T094302Z` 完成 4 题、32 个独立宏组、256 回答；ρ=0.9209、qρ=0.9024，探索性区间跨 1，INCONCLUSIVE。收益不稳定的机制原因尚未定位，不能归因于评分严格。已完成 CPU 原评分/权重/统计复算和逐题敏感性分析，见[实验报告](../reports/trend_pilot/REPORT.md)及[详细诊断](../reports/trend_pilot/ANALYSIS.md)。139 tests、lint、exact 通过。尚未证明训练收益，不自动扩样，定时监控按用户要求暂停。

以下是保留的 v4 历史状态。

# 当前状态

更新时间：2026-09-07 约 04:58 UTC。当前两小时周期为 03:50:25–05:50:25 UTC。最强证据层级是完整 Dev 固定候选银行、两题全参数梯度与 Dev 功能步长校准；pilot 仍为 INCONCLUSIVE，没有 GO 或训练授权。

## 已完成并有证据

- dependent feedback v4 是主线；原始 v4、Pro 建议和旧 dependent-rollouts 档案保持不变。
- MBPP+ 固定任务银行包含 363 题；Dev/C/D/evaluation 为 16/32/32/32，每题 8 个 hash 选择测试。任务 SHA256 为 a39d757611cb58d9d7f48f49e0920f0be1b4c6dd7a93cd9dcda28bf6f37746b4。
- 本地 CPU/随机模型检查为 94 passed；v4 与 legacy exact verifier PASS。
- Qwen/Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28 的 32-token FP32 全参数 GPU-0 plumbing 已通过。
- Dev 前两题固定银行完成：32 回答、3234 token、0 截断、0 解析失败；S-I 全参数梯度 L2 为 11.98609163397219，远端 gradient SHA256 为 300dd4f62eb6d13f80e500d5b4769e3ccde6a8a828a0f79444bec0f8933df74a。
- 完整 Dev 固定银行完成并评分：16 prompts、32 groups、256 responses、24,515 generated tokens、0 truncation、10 parse failures、178 suite passes。max token-logp error 为 1.9355466852211123e-4；K2/K3 二值阴性对照最大误差为 1.1102230246251565e-16。
- 完整 Dev 的 32 组中，6 组出现非零候选信用差，分布于 4 个 prompt；其中 3 组在候选信用空间有大于 1e-12 的正缩放残差。这只是固定银行描述，不是功能梯度、总体均值或显著性结论。

主要回执：

- reports/dev_all16_credit_2h.json，文件 SHA256 0913ea6350da1276fe2a95e953f89280ba5c6cf8fe06161545544425fa087d9f，bank SHA256 65bdaaa5e55b0ad9a0021b7e7ec7f437efceb3d3724bffc22a8be7fda3de0d29。
- reports/dev_first2/gradient/receipt.json。
- reports/dev_first2/step_calibration/receipt.json。
- reports/gpu0/qwen7b_fp32_cpuemb.json、reports/validation/local_tests.log、reports/validation/exact.log。

## Dev 功能步长校准：未通过

预声明读出使用 Mbpp/14 与 Mbpp/577 的 canonical function 对 constant-None stub log-odds。S-I 功能斜率为 -3470.964737065184 和 -1226.121338824465；独立臂斜率为 -39713.63212926152 和 10858.585942747843。功能空间最佳正缩放系数为 1.0734658925229812，残差为 2098.1450669848523；该量仍是 Dev 描述，不是显著性检验。

| step | 预测 S-I 变化 | 实测 S-I 变化 | relative residual |
|---:|---|---|---:|
| 1e-4 | [-0.3470965, -0.1226121] | [-0.7096393, -0.2336031] | 1.0299787 |
| 5e-5 | [-0.1735482, -0.0613061] | [-0.2990813, -0.0896406] | 0.6991867 |

残差随步长减半而下降，但两个步长都未达到可接受的线性区间。因此没有冻结可用 eta，原计划的 C/D 参数更新与局部 consequence 分支已取消。该回执明确记录 independent reward consequence 为 NOT_MEASURED、independent target slopes 为 null、pilot decision 为 INCONCLUSIVE。

下一次参数更新前，必须在 Dev 预声明并测试小于 5e-5 的 eta，验证 eta/eta/2 的残差规律；不能依据 C/D 结果选择步长。

## 当前运行：C8/D4 投影诊断

已启动预声明的小规模确认诊断：

- C：冻结的 8 个 prompt，每题 1 个 K=8 原始组。
- D：冻结的 4 个独立 prompt，每题 1 个 K=8 原始组。
- 只计算已冻结功能读出和独立 D 目标投影。
- 不执行参数更新，不声称局部 reward consequence。
- C/D 原始银行与评分已经封存：C64、D32 回答；C 全参数 S−I L2=3.9798321396317893，receipt 见 reports/cd_prefix_2h/gradient/。功能与目标投影已完成：D点估计+81.992733，单条回答敏感；不构成GO。

该缩小诊断用于检查固定功能与独立目标投影是否值得继续，不替代完整 C32/D32、不提供完整双银行不确定性，也不修复 Dev 步长失败。

## 存储与保留

Dev 前两题 30,462,505,048-byte 梯度缓存已于 04:55:55 UTC 删除，原始银行、配置、源码、hash、函数与测量结果保留，缓存可重算。见 reports/dev_first2/gradient/retention.json。C 梯度已保存并封存。

## 尚未建立

- 完整 C32/D32 总体效应与稳定奖励收益。
- 可接受的 Dev 线性步长、独立 reward consequence 或 eta/eta/2 局部后果。
- 自然代码反馈中的总体 S-I 信用差异、完整 C/D 置信区间或 pilot GO。
- 在线训练、rubric 场景、第二模型复现或论文级实证结论。

reports/pilot_result.json 记录已运行阶段，判决保持 INCONCLUSIVE。完整 Dev fixed-bank credit、两题梯度和失败的步长校准都不能单独解锁训练。

## 本周期最终方向结果

D目标点估计+81.99273267；单条贡献+85.11101439，leave-one-response-out、prompt等权诊断为−4.72563587。主估计保留全部回答。没有完整双银行CI，判决INCONCLUSIVE。完整报告：[两小时实验报告](../reports/cycle_2h_REPORT.md)。
