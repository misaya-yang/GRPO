# Dependent feedback v4 实验计划

## 研究目标

固定 actor checkpoint、IID 回答和逐回答反馈边际，只改变外生评分配置的组内关联：

- S / shared：整组共享同一测试或 rubric 配置。
- I / independent：每条回答独立抽取配置。
- RLOO：在外生 verifier coupling 下的均值不变阴性对照。

首轮用冻结 MBPP+ 测试矩阵判断：自然代码反馈下，S 与 I 的标准化平均信用是否可分辨；差异是否超出单一全局正缩放；它是否预测独立 D 目标和同初态局部更新。方向反转不是必须结果，只有 norm/count 变化也不足以 GO。

## 固定合同

- Actor 回答 IID；配置向量与完整回答向量独立，不按回答或奖励选择测试。
- 每题固定 8 个测试，配置均匀；候选先生成，再在 CPU 逐测试执行并保存完整 verdict。
- K=8、每题 2 个独立 actor 组；全同奖励、解析失败、候选超时和截断均保留。
- temperature=1、top_p=1、top_k=0、无 repetition penalty；实际 EOS 计入 score，padding 不计，截断不补 EOS。
- ddof=0，epsilon 为根号外 1e-6；detached advantage 乘 response-token logp 之和。
- 主机制固定 checkpoint、全参数、FP32、eager、无 clipping/KL/长度平均/off-policy reuse。
- prompt 是推断单位；C/D 独立。虚拟 mask 和精确积分不增加样本量。
- 每阶段使用新目录和固定预算 ledger。失败保留；基础设施失败不记成候选失败。

模型名称不是科学合同。优先复用可用且任务能力合适的权重。当前使用已有 Qwen2.5-7B-Instruct，revision a09a35458c702b33eeacc393d103063234e8bc28；如果资源证据要求更换模型或存储布局，冻结新配置并按实际参数化限定结论。

## 当前执行顺序

| 优先级 | 阶段 | 规模与动作 | 完成条件 / 当前状态 |
|---:|---|---|---|
| 0 | CPU 合同 | 87 个 CPU/随机模型测试；v4 + legacy exact verifier | PASS；不等于预训练结果 |
| 1 | GPU-0 方法 probe | 1 个 Dev prompt、1 条回答、32 新 token；生成、重评分、全参数梯度 | PASS；receipt 已回传 |
| 2 | bounded Dev 前段 | 冻结 Dev 排序的前 2 题 × 2 组 × K=8，最多 32 回答、每条 cap 384 | COMPLETE fixed bank：32回答/3234tokens；不按结果选题 |
| 3 | 完整 Dev fixed bank | 16 prompts × 2 groups × K=8 | COMPLETE：256 responses / 24,515 tokens；仅 fixed-bank credit |
| 4 | Dev 功能步长 | 冻结 2 个有符号读出；eta=1e-4 与 5e-5 | FAILED GATE：relative residual 1.02998 / 0.69919；无 accepted eta |
| 5 | C8/D4 投影诊断 | C 8 prompts、D 4 independent prompts、每题 1 个 K8 组 | RUNNING；只做 frozen-function 与 independent-D target projection，无参数更新 |
| 6 | 更小 eta Dev 校准 | 在看 C/D 投影之外预声明 eta<5e-5 与半步 | PENDING；通过后才恢复 local consequence |
| 7 | 完整 C/D 确认 | 原计划 C32/D32 与 prompt 级双银行不确定性 | PENDING；C8/D4 不替代该阶段 |
| 8 | 判决与后续 | 写入 evidence-bearing GO/STOP/INCONCLUSIVE | 当前 INCONCLUSIVE；训练锁定 |

GPU-0 v1 在生成前暴露 Transformers 5.15 chat-template BatchEncoding 接口差异。v2 已生成 32 token 且 score tolerance 通过，但全参数 FP32 backward 因显存不足中止。v3 保持模型、FP32 和全参数范围，仅将输入 embedding 与 post-accumulate gradient 存到 CPU，已完整通过。上述修复改变接口兼容与存储位置，不改变 estimand。

## GPU-0 实测基线

Receipt：reports/gpu0/qwen7b_fp32_cpuemb.json。

- 32-token generation：2.8519 s。
- 全参数 FP32 gradient：29.4187 s。
- max token error：3.9583e-6，小于 1e-5。
- peak GPU allocation：32,080,404,992 bytes。
- 自然 S/I 效应、局部后果与 pilot decision：仍未测量。

## 当前阶段决定

完整 Dev fixed bank 已完成，但它不提供独立确认。两题 Dev 功能校准表明 eta=1e-4 和 5e-5 都不在已接受的线性范围。因此取消本轮 C/D 参数更新，保留固定功能与独立 D 目标投影；这避免用未校准步长制造 consequence 结论。

C8/D4 在 03:50:25–05:50:25 UTC 两小时周期内按预声明子集运行：C 8 题、D 4 题，每题仅 1 个 K8 原始组。它是低成本方向诊断，不是完整 C32/D32，也不能用来选择有利 prompt 或反向调整 Dev 步长。下一次局部更新前必须在 Dev 预声明并验证小于 5e-5 的 eta。

## 资源与停止规则

不得并行启动模型、mask、K、精度或任务网格。单次失败只允许做与已观测原因直接对应的改动，并把改动与失败 receipt 关联。

当前硬件不是 v4 原估算中的 5090/Pro 6000，因此不沿用旧吞吐承诺。每阶段由 ledger 冻结 cap。到 cap 时停止并报告已完成部分；资源或统计精度不足为 INCONCLUSIVE，并最多给出一次基于实际数据的定量扩样或替代配置建议。

约 30 GB 的 Dev 前两题 S-I 全参数梯度 cache 可以从冻结 raw banks 重算。若为 C gradient 释放空间，必须先复制并核验 receipt、functions、配置与 raw candidate/scored banks，随后用独立运维回执记录删除目标和释放空间；不可删除唯一原始证据。

## 判决门

GO 必须同时满足：

1. implementation：采样与重评分一致，EOS/reduction/奖励边际/DP 合同通过；K=2/3 与 RLOO 阴性对照成立。
2. credit：自然代码反馈下，S-I 在预定有符号功能方向或独立目标斜率上有稳定、可分辨的量级。
3. non_scalar：差异不能由单一全局正尺度解释；报告它来自跨 prompt 重加权、同题行为差异或两者。
4. local_steps：共同 eta 与 eta/2 的实际局部变化与一阶预测基本相容，且不由少数解析/执行异常驱动。

沿用 v4 的参考门槛：相对已分辨参考功能变化约 10%，或 Dev 预先冻结的绝对阈值。分母接近零时不报告不稳定比例。

STOP 用于合同正确且区间足够窄时排除实践量级，或效果只剩 norm/全局正缩放、只在选择性弱 verifier 中出现。吞吐不足、参考近零、组数不足、执行器失败或 C/D 不确定性未传播均为 INCONCLUSIVE。reports/pilot_result.json 按真实阶段更新；四门证据齐全前判决保持 INCONCLUSIVE。

## GO 后范围

在线训练只比较预先冻结并匹配预算的原生 shared standardized、independent standardized 与 shared RLOO。先做单个 Dev 学习性检查，再决定种子复制；训练结束仍需共同 IID 评价。第二模型家族和 rubric 场景只在首轮通过且有独立预算时开始，并分别登记新假设。
