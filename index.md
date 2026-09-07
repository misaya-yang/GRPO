# 项目索引

## 当前结论

主线为 dependent feedback v4：回答保持 IID，只改变外生评分配置在组内 shared 或 independent。代码与 CPU 机制检查已具备；真实预训练模型 pilot 尚未形成 GO/STOP 结论。最新状态见 [docs/STATUS.md](docs/STATUS.md)。

## 合同与入口

| 文件 | 用途 |
|---|---|
| [agent.md](agent.md) | 核心约束、证据边界、模型与执行环境规则 |
| [AGENTS.md](AGENTS.md) | Agent 自动发现入口 |
| [README.md](README.md) | 项目详情及 Mac/家用 PC 环境 |
| [实验计划](docs/experiments/PLAN.md) | v4 分阶段优先级、预算边界、GO/STOP/INCONCLUSIVE |
| [执行手册](docs/experiments/RUNBOOK.md) | 与当前 CLI 一致的原生运行顺序 |
| [冻结清单](docs/experiments/MANIFEST.md) | 当前模型、数据、配置与待补回执 |
| [实验机记录](docs/operations/EXPERIMENT_HOST.md) | 服务器、资产、网络与清理记录 |

## 理论主线与原始来源

| 文件 | 状态 / 内容 |
|---|---|
| [dependent_feedback_v4.md](docs/theory/dependent_feedback_v4.md) | 当前原始研究合同；保持原字节 |
| [feedback_v4_pro_review.md](docs/theory/feedback_v4_pro_review.md) | Pro 审核建议；保持原字节 |
| [FEEDBACK_THEORY.md](docs/theory/FEEDBACK_THEORY.md) | 可修订理论说明：一般不变性定理、ANOVA 交互、大组极限、统计边界 |
| [feedback_sources.json](docs/theory/feedback_sources.json) | 原始来源与验证器 hash |
| [PROVENANCE.md](docs/theory/PROVENANCE.md) | 来源与归档说明 |
| [dependent_rollout_research_dossier.md](docs/theory/dependent_rollout_research_dossier.md) | 旧回答依赖主线，仅作历史对照 |
| `docs/archive/` | v4 迁移前文档快照，不作为当前状态 |

## 当前实现

| 文件 | 功能 |
|---|---|
| `src/reward_coupling/advantages.py` | RLOO、population-std 与优势合同 |
| `src/reward_coupling/expectation.py` | 二值 Poisson-binomial、有限支持与 rubric mask 精确积分 |
| `src/reward_coupling/theory.py` | 有限银行不变性、ANOVA/LP 与二值总体公式 |
| `src/reward_coupling/bank.py` | 完整组校验、hash、S/I/RLOO 权重及 C/D 独立性 |
| `src/reward_coupling/sample.py` | IID 祖先采样、逐 token logp 与候选银行 |
| `src/reward_coupling/execute_tests.py` | 原生逐测试 worker；Docker 可选 |
| `src/reward_coupling/feedback.py` | 代码测试/缓存 rubric 评分与权重积分 |
| `src/reward_coupling/gradient_audit.py` | shared/independent 等全参数梯度和 Dev 读出 |
| `src/reward_coupling/local_steps.py` | 同一初态、共同 `eta`/`eta/2` 与独立 D 评价 |
| `src/reward_coupling/statistics.py` | prompt bootstrap、双银行、正缩放与 pilot 门禁 |
| `src/reward_coupling/training.py` | GO 后原生 shared/independent/full/RLOO 在线训练 |
| `src/reward_coupling/judge.py` | response-local、本地 pinned rubric judge |
| `src/reward_coupling/budget.py` | 跨命令硬墙钟预算 ledger |
| `src/reward_coupling/contract.py` / `cli.py` | 配置预检与 `preflight/collect/score/integrate/audit/freeze-readout/branch/train` CLI |
| `scripts/prepare_mbpp_feedback.py` | actor 生成前冻结 MBPP+ 子集与测试 |
| `scripts/gpu_probe.py` | 单回答、32-token 预训练 score/全梯度 plumbing probe |
| `scripts/run_pilot.py` | 单阶段预算 supervisor，不自动发起矩阵 |
| `scripts/validate_exact.py` | v4 与 legacy CPU 精确验证器 |

## 数据、配置与证据

| 路径 | 当前状态 |
|---|---|
| `data/feedback/mbpp_pilot.jsonl` | 363 条冻结任务；Dev/C/D/evaluation 各 16/32/32/32，其余 train |
| `data/feedback/mbpp_pilot.jsonl.receipt.json` | 数据、源和 decoder hash；参考程序现场检查仍是前置项 |
| `configs/feedback/pilot_code.json` | v4 模板；仍含旧 Coder-1.5B placeholder，不能直接作为已冻结运行配置 |
| `configs/feedback/pilot_rubric.json` | 预注册二级场景模板；代码场景通过前不启动 |
| `reports/validation/local_tests.log` | Mac `aidemo`：94 passed；随机/CPU plumbing 证据 |
| `reports/validation/exact.log` | 指向 `runs/exact/20260907T025718Z.json`；v4 + legacy CPU verifier PASS |
| `reports/gpu0/qwen7b_fp32_cpuemb.json` | Qwen2.5-7B-Instruct 32-token 生成、score point 与 FP32 全参数梯度 plumbing PASS；不含自然 S/I 效应 |
| `reports/dev_all16_credit_2h.json` | 完整 Dev fixed bank：16 prompts、32 groups、256 responses；只含候选信用诊断 |
| `reports/dev_first2/gradient/receipt.json` | 前两题 S-I 全参数梯度回执；L2 11.98609163397219，远端 gradient SHA256 300dd4f... |
| `reports/dev_first2/step_calibration/receipt.json` | eta 1e-4 / 5e-5 relative residual 1.02998 / 0.69919；没有 accepted linear step |
| `reports/pilot_result.json` | `NOT_RUN` / `INCONCLUSIVE`；所有四门均无预训练证据 |
| `reports/remote_cleanup_20260907.json` | 307 项旧结果/代码清理完成，保留模型资产 |
| [docs/CLAIMS.md](docs/CLAIMS.md) | 声明到证据等级对应表 |

完整 Dev fixed bank 已完成。Dev 两题的功能步长校准没有找到可接受的线性 step，因此取消当前 C/D 参数更新；下一次更新前需预声明更小的 Dev eta。当前两小时周期只运行预声明 C8/D4 的 frozen-function 与 independent-D target projections，投影已完成，点估计正但单条回答敏感。大型候选银行、梯度和局部步长结果放在各运行目录并以 receipt/hash 索引。

## 历史实现

`src/dependent_rollouts/`、旧 `configs/pilot_*.json`、`paper/main.tex` 及相关 W/C/X 记录属于上一条回答依赖研究线。可复用其中的通用 artifact/intervention 工具，但不得把旧 W/C/X 结果当作 v4 的 S/I 预训练证据。

## 2026-09-07 首轮实测

- [GPU0 回执](reports/gpu0/qwen7b_fp32_cpuemb.json)：现有 Qwen7B 的32token采样与全参数FP32梯度已通过。
- [完整回答数值校准](reports/numerics/dev_first2.json)：逐前缀/整序列 logp 误差测量。
- [Dev 前两题报告](reports/dev_first2/REPORT.md) / [信用结果](reports/dev_first2/credit.json)：32回答固定银行已完成；尚非总体机制结论。
- [完整 Dev 信用报告](reports/dev_all16_credit_2h.json)：256回答 fixed-bank credit；不含全参数总体梯度或独立确认。
- [Dev 步长回执](reports/dev_first2/step_calibration/receipt.json)：两个 step 都未通过线性残差门，pilot 保持 INCONCLUSIVE。
- [服务器环境下 CPU 测试](reports/validation/remote_cpu_tests_final.log)：94 passed；本地与服务器版本不同，分别记录。

## 当前两小时周期

- [两小时实验报告](reports/cycle_2h_REPORT.md)
- [周期计划和截止时间](reports/cycle_2h_20260907.json)
- [C/D 银行验证](reports/cd_prefix_2h/bank_validation.json)
- [C 固定银行信用](reports/cd_prefix_2h/C_credit.json) 与 [D 固定银行信用](reports/cd_prefix_2h/D_credit.json)
- [Dev 梯度缓存释放回执](reports/dev_first2/gradient/retention.json)

- [原始/评分银行本机备份索引](reports/raw_bank_backup_index.json)：数据位于 `runs/remote_archive_2h/`，每个文件独立 hash；`runs/` 不纳入 Git。

- [下一轮 Dev 较小步长计划](reports/next_dev_calibration_plan.json)：已准备，未运行。

- [C→D 完整方向回执](reports/cd_prefix_2h/direction/receipt.json)及[敏感性诊断](reports/cd_prefix_2h/direction/sensitivity.json)
- [本周期 pilot 判决](reports/pilot_decision_2h_20260907.json)：INCONCLUSIVE

## v5 离线审查

- [v4受挫原因与v5理论审查](reports/v5_review/ANALYSIS.md)：ANALYZED；未重启服务器，未改变主线。含给Pro的五项优先问题。
- [独立分层重配对有限枚举](reports/v5_review/verify_reblocking.json)：CPU有限例子，不是真实模型收益。
