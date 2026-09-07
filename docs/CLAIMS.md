# Dependent feedback v4 声明与证据

证据等级：

- derived：在 v4 或理论说明中给出推导；不等于同行评审或原创性确认。
- cpu-verified：有限枚举、DP 或数值线性代数检查。
- plumbing：随机模型或单点真实 checkpoint 接口检查。
- measured fixed-bank：冻结候选银行上的描述性测量。
- measured confirmation：按预注册 C/D 合同得到并传播相应不确定性的确认。
- NOT_RUN：尚无对应实验。

不得跨等级升级；每次升级必须列出不可变 receipt/hash 和实际模型范围。

| 声明 | 当前等级 | 证据与边界 |
|---|---|---|
| 二值 standardized 下 G_S=E_U[w(p_U)g_U]、G_I=w(Ep_U)E g_U | derived + cpu-verified | dependent_feedback_v4.md §3、FEEDBACK_THEORY.md 与 exact receipt；有限检查不证明自然 LLM 量级 |
| 外生 verifier coupling 下 RLOO 平均更新不变 | derived + cpu-verified | v4 §5；不扩展到方差、Adam 状态或有限步轨迹 |
| K=2/3 二值 standardized 与 RLOO 逐样本退化，K≥4 可有交互 | derived + cpu-verified | exact verifier；连续或多值奖励不继承该退化 |
| 相同逐回答反馈边际不足以确定一般 advantage 的期望 | derived + cpu-verified | FEEDBACK_THEORY.md 的有限奖励刻画与有限表检查 |
| 连续奖励的大组 S/I 极限可以不同 | derived | 要求有界奖励、epsilon>0 与 score 可积性 |
| v4 反向例与连续 K=2 例存在 | derived + cpu-verified | 存在性反例，不是自然代码反馈结果 |
| MBPP+ 任务/测试在 actor 前冻结 | cpu-verified artifact | 363 题、每题 8 测试；Dev reference receipt 已通过 |
| 当前实现通过 CPU/随机模型检查 | plumbing PASS | reports/validation/local_tests.log：94 passed；exact logs PASS |
| Qwen2.5-7B 的 32-token 生成、score point 与 FP32 全参数 gradient | plumbing PASS | reports/gpu0/qwen7b_fp32_cpuemb.json |
| Dev 前两题全参数 S-I gradient | measured fixed-bank | L2 11.98609163397219；gradient SHA256 300dd4f...；范数不是功能效果或收益 |
| 完整 Dev 候选与评分银行 | measured fixed-bank | reports/dev_all16_credit_2h.json：16 prompts、32 groups、256 responses、24,515 tokens、0 truncation、10 parse failures、178 suite passes |
| 完整 Dev 存在候选信用差 | measured fixed-bank | 6/32 groups 非零，覆盖 4/16 prompts；3 groups 有非零正缩放残差。仅条件候选信用描述 |
| Dev 功能空间 S/I 非全局正缩放 | measured Dev description | 两个冻结读出上的 positive scale 1.07347、residual 2098.145；不是确认集显著性 |
| eta=1e-4 或 5e-5 位于已接受线性区间 | FAIL on Dev calibration | relative residual 1.02998 / 0.69919；没有 accepted eta |
| eta/eta/2 局部 consequence 已确认 | NOT_RUN | C/D 参数更新已取消；需先在 Dev 预声明更小 eta 并通过残差检查 |
| C8/D4 固定功能与独立目标投影 | measured preliminary direction | 预声明 8 个 C prompt、4 个独立 D prompt、每题 1 个 K8 组；只做投影，不做更新 |
| 自然代码反馈下 S-I 有总体可分辨差异 | NOT_RUN | C8/D4 点估计正但不稳健，规模不提供完整 C32/D32 推断 |
| Pilot GO/STOP | INCONCLUSIVE | step gate 未通过；C/D 确认未完成；reports/pilot_result.json 按阶段记录，判决 INCONCLUSIVE |
| 在线训练或泛化收益 | NOT_RUN | 需要 evidence-bearing GO 与共同 IID held-out 评价 |
| Rubric/LLM judge 与第二模型复现 | NOT_RUN | 代码主场景通过后另行预注册；不得补选阳性 |

## 当前解释

完整 Dev 证明冻结自然回答银行中确实存在评分关联导致的候选信用差，并提供了效应稀疏度和数值范围。它没有把这种差异提升为总体功能梯度或独立目标改善。

Dev 步长结果表明当前尝试的两个 eta 都太大，无法支持线性局部解释。残差随减半改善只能说明继续向更小 eta 校准有依据；不能把趋势外推成已经通过。

C8/D4 是资源受限的投影诊断。它可以提供后续设计信息，但没有参数更新、完整 prompt 规模或完整双银行区间时，不构成 local_steps 门、总体机制结论或 GO。
