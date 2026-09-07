# Dev 前两题：自然反馈固定银行预实验

模型：复用 Qwen2.5-7B-Instruct，revision `a09a35458c702b33eeacc393d103063234e8bc28`。
FP32、eager、全参数；input embedding 与累积梯度放 CPU，未使用 LoRA。

## 预定工作量与数据

冻结 Dev 顺序的前两题 `Mbpp/14`、`Mbpp/577`，每题 2 个独立 K=8 组；
每条响应 cap384，保存实际 EOS。共32回答、3234 response tokens，0截断、0解析失败。
题目和每题8个测试均在 actor 生成前按 hash 选定，未按结果换题、补样或删组。

候选银行 SHA256：`7f5853a9ec8e1551ba0053dd76470e751b6d6c1dbe7801eb2b52cc8cfdb12229`。
评分银行 SHA256：`9a82c49920be3e2975d0416bf707a90cb80cebcbbefb702daf5e7601629bd0c6`。

## 固定银行结果

| prompt / group | 平均测试奖励 | suite pass率 | ||A_S−A_I|| | 最佳正缩放残差（候选信用空间） |
|---|---:|---:|---:|---:|
| Mbpp/14 / 0 | 1 | 1 | 0 | 参照为零 |
| Mbpp/14 / 1 | 1 | 1 | 0 | 参照为零 |
| Mbpp/577 / 0 | 0.34375 | 0 | 0.0896213 | 0.0743474 |
| Mbpp/577 / 1 | 0.34375 | 0 | 0.0192919 | 约3.4e-17 |

二值 K=2/3 固定子组阴性对照最大误差约8.33e-17。子组仅用于代数核验，
不计作新的独立样本。16/32个回答通过整个已冻结8测试集，不称为完整 MBPP+ benchmark accuracy。

一组出现非标量候选信用差异，另一组仅正缩放。全通过题无更新信号。
这些观察仅对本次固定银行成立；不能推广为总体均值差、功能方向、损害或训练收益。

## 数值和执行过程

32-token GPU0 的误差上限3.96e-6，不代表完整回答的上限。
两条完整 Dev 校准回答测得2.35e-5/3.98e-5；收集第27条时出现1.03e-4。
该错误发生在查看奖励之前。程序保存了前26条和触发检查的第27条，
随后冻结2e-4容差、从同一组seed/同一候选继续，仅生成剩余5条。
完整银行最大误差为1.19764e-4。旧尝试与断点输入hash均留档，不冒充一次无故障运行。

这是 FP32 概率实现的数值核验，不声称逐位一致或理想实数算法的精确实现。
完整序列 teacher forcing 与逐前缀生成的 kernel 求和次序不同，需要记录误差。

## 梯度阶段

固定银行 S−I 全参数诊断已完成并核验gradient/manifest SHA256。全参数梯度L2范数11.98609163397219，计算与保存耗时609.9558秒，最大token误差1.19764e-4。该范数不代表功能效应大小或学习收益。首个尝试暴露零权重行仍建立/保留 autograd 图，
已修正为零权重 score 核验使用 no_grad、每条结束释放图，增加回归测试。
保留失败日志，不把它解释成科学阴性。

第二个梯度尝试还暴露了CPU匿名内存副本问题：宿主机显示503GiB，但实例cgroup上限62GiB。
当前改为一次全局加权累积，保留两个prompt等权，避免逐prompt梯度与全局梯度各占约30GB。
激活可转存CPU，参数存储保留原计算路径；前两次失败尝试不算有效结果。

最终梯度回执：`gradient/receipt.json`；全参数缓存已在完成校准后释放，原文件 SHA256 为 `300dd4f62eb6d13f80e500d5b4769e3ccde6a8a828a0f79444bec0f8933df74a`。保存了实际执行源码快照。

## 功能步长校准

在查看确认集之前，Dev 预声明了两个有符号读出：`Mbpp/14` 与 `Mbpp/577` 的 canonical function 相对 constant-None stub 的 log-odds。定义文件为 `step_calibration/functions.json`，SHA256 `3b97ca1f52b9a4f7a72bc2fe2e1e9e8015e577bdc085c0bb0038081115a7fa13`。

| step | 预测 S−I 读出变化 | 实测 S−I 读出变化 | relative residual |
|---:|---|---|---:|
| 1e-4 | [−0.3470965, −0.1226121] | [−0.7096393, −0.2336031] | 1.0299787 |
| 5e-5 | [−0.1735482, −0.0613061] | [−0.2990813, −0.0896406] | 0.6991867 |

残差随步长减半而下降，但两个值都未进入可接受的线性区间。没有冻结可用于确认的 eta，因而取消本轮 C/D 参数更新和 reward-consequence 分支。回执中的 `independent_reward_consequence` 为 `NOT_MEASURED`，`independent_target_slopes` 为 null，不能把这次校准写成局部后果已验证。

下一次参数更新前必须在 Dev 预声明并测试小于 5e-5 的 eta 及其半步，继续检查残差规律。不能用 C/D 投影结果反向选择步长。完整回执为 `step_calibration/receipt.json`，文件 SHA256 `e0cdaad660344d6b61708873ee6bdf1b7ea6c635c13896c3b3b0e8f3d7f65c38`。

## 完整 Dev 固定银行

同一冻结 Dev 划分已扩到 16 prompts、32 original groups、256 responses，共 24,515 generated tokens。0 条截断，10 条解析失败，178 条通过全部 8 个冻结测试；最大 token-logp error 为 1.9355466852211123e-4。K=2/3 二值阴性对照最大误差为 1.1102230246251565e-16。

32 个组中 6 组有非零候选信用差，涉及 4 个 prompt；其中 3 组在候选信用空间有大于 1e-12 的最佳正缩放残差。这说明效应在该固定银行中稀疏且异质。它仍不是全参数总体梯度、prompt 级显著性或独立目标后果；该报告明确标记 `pretrained_gradient_difference=NOT_MEASURED_HERE`、`independent_confirmation=false`、`pilot_decision=INCONCLUSIVE`。

报告：`../dev_all16_credit_2h.json`；文件 SHA256 `0913ea6350da1276fe2a95e953f89280ba5c6cf8fe06161545544425fa087d9f`；scored bank SHA256 `65bdaaa5e55b0ad9a0021b7e7ec7f437efceb3d3724bffc22a8be7fda3de0d29`。

## 当前 C8/D4 诊断边界

03:50:25–05:50:25 UTC 周期内已启动预声明的 C8/D4 诊断：C 使用 8 个冻结 prompt，D 使用 4 个独立 prompt，每题 1 个 K=8 原始组。只计算冻结功能投影和独立 D 目标投影，不做任何参数更新。当前尚无结果回执。

这一诊断不能替代完整 C32/D32 或双银行不确定性，也不能通过 local_steps 门。当前状态保持 **INCONCLUSIVE，非 pilot GO，不启动训练**。

## 存储保留边界

Dev 前两题 S-I 的 30,462,505,048-byte 全参数梯度缓存已于 04:55:55 UTC 释放，用于容纳 C 梯度。原始银行、配置、源码、函数、测量结果与原文件 hash 已保留，缓存可重算；删除回执见 `gradient/retention.json`。
