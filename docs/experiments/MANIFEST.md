# Dependent feedback v4 冻结清单

本文件区分已经冻结的科学输入、当前运行选择和仍需 receipt 的值。后续变更追加带日期的新块，不覆盖旧回执。任一 PENDING 都不能由口头状态替代。

## 原始理论与数据

| 项目 | 冻结值 | 状态 |
|---|---|---|
| v4 原文 | docs/theory/dependent_feedback_v4.md；SHA256 48d4f913e5d9cc096e19ad3e4fed22dce216d352bb4d39b0c5efd01acd9dc63a | FROZEN |
| 可修订理论说明 | docs/theory/FEEDBACK_THEORY.md；SHA256 73cdc59d6ab910afbae0a147a66c06a4f73e85acafab9d89274f1a4b3877e126 | 当前快照 |
| 任务银行 | data/feedback/mbpp_pilot.jsonl；SHA256 a39d757611cb58d9d7f48f49e0920f0be1b4c6dd7a93cd9dcda28bf6f37746b4 | FROZEN before actor |
| MBPP+ 源 | SHA256 b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad | FROZEN |
| input decoder | SHA256 186e09b7b14dcf12ed259dfca1d6644a1c266c1bbac41c917fd3c0449a734b24 | FROZEN |
| 数据划分 | Dev 16、C 32、D 32、evaluation 32；其余 train；seed 20260906 | FROZEN |
| Dev 银行 | 冻结 16 题；每题 2 组 × K=8；不看结果选题 | COMPLETE：256 responses |
| 每题测试 | hash 选择 8 个测试；预声明 special oracle 排除；参考程序逐题现场检查 | 矩阵已冻；16 个 Dev reference PASS |

## 方法合同

| 项目 | 冻结值 |
|---|---|
| Actor | IID ancestral；temperature=1、top_p=1、top_k=0、repetition penalty 1 |
| 分组 | K=8；每题 2 个独立原始组；完整 Dev/C/D 每题 16 回答 |
| 反馈 | 代码主场景：8 测试均匀选 1；S 共享、I 独立、RLOO 阴性对照 |
| 标准化 | population std（ddof=0）；根号外 epsilon=1e-6；全同奖励组保留且优势为零 |
| Score/loss | 实际 EOS 计入；padding 不计；截断不补 EOS；detached advantage × token-logp sum |
| 主机制 | on-policy、FP32、eager、全参数、identity map；无 clipping、KL reward、长度平均、off-policy reuse |
| 数据完整性 | 完整 actor 组；不按奖励筛选/补样；每阶段新输出目录；receipt/hash 验证 |
| 执行器 | dedicated host 原生逐测试 Python worker；Docker 可选 |
| 判决 | implementation、credit、non_scalar、local_steps 四门全带 hash 证据才可能 GO |

## 2026-09-07 GPU-0

| 项目 | 当前值 | 状态 |
|---|---|---|
| Actor model | Qwen2.5-7B-Instruct | 已使用已有权重 |
| Model revision | a09a35458c702b33eeacc393d103063234e8bc28 | FROZEN |
| Tokenizer revision | a09a35458c702b33eeacc393d103063234e8bc28 | FROZEN |
| Chat template hash | 从 candidate-bank manifest 读回并登记 | PENDING index update |
| 参数范围 | all actor parameters | FROZEN |
| Forward / gradient dtype | FP32 | FROZEN |
| 梯度存储 | CPU post-accumulate storage；输入 embedding 驻 CPU | FROZEN for successful probe |
| Probe workload | 1 个 Dev 题、1 条回答、最多 32 新 token | FROZEN |
| GPU-0 v1 | chat-template BatchEncoding 接口失败，生成前中止 | FAILED_INFRASTRUCTURE |
| GPU-0 v2 | 32 token 与 score tolerance 通过；full gradient OOM | PARTIAL |
| GPU-0 v3 | reports/gpu0/qwen7b_fp32_cpuemb.json | PASS_PLUMBING |
| v3 timing | generation 2.8519 s；full gradient 29.4187 s | MEASURED |
| v3 numerics | max token error 3.9583e-6；peak GPU 32,080,404,992 bytes | PASS for probe contract |
| 自然 coupling effect | 未测 | NOT_MEASURED |
| Pilot decision | reports/pilot_result.json: NOT_RUN / INCONCLUSIVE | FROZEN current status |

ModelScope 的 Coder 下载在约 2 GB partial 时已停止，未进入任何运行配置或证据。使用 7B Instruct 不是理论变更；后续换模型需新增运行块并保持同一 feedback estimand。

## 2026-09-07 Dev 实测

| 项目 | 冻结值 | 状态 |
|---|---|---|
| Prompt selection | data/feedback/mbpp_pilot.jsonl 的 16 个冻结 Dev prompt | FROZEN before outputs |
| Workload | 16 prompts × 2 original groups × K=8 = 256 responses | COMPLETE |
| Token cap | 384 per response | FROZEN |
| Model contract | 与成功 GPU-0 相同 revision、FP32、eager、全参数；CPU input/gradient storage | FROZEN |
| Fixed-bank receipt | reports/dev_all16_credit_2h.json；file SHA256 0913ea6350da1276fe2a95e953f89280ba5c6cf8fe06161545544425fa087d9f | COMPLETE |
| Scored bank | SHA256 65bdaaa5e55b0ad9a0021b7e7ec7f437efceb3d3724bffc22a8be7fda3de0d29 | FROZEN |
| Counts | 24,515 tokens；0 truncation；10 parse failures；178 suite passes | MEASURED fixed-bank |
| Numerics | max token-logp error 1.9355466852211123e-4；K2/K3 control 1.1102230246251565e-16 | PASS current tolerance |
| Pretrained gradient difference | 完整 Dev receipt 标记 NOT_MEASURED_HERE | NOT_MEASURED |

## 2026-09-07 Dev step calibration

| 项目 | 冻结值 | 状态 |
|---|---|---|
| Definitions | reports/dev_first2/step_calibration/functions.json；SHA256 3b97ca1f52b9a4f7a72bc2fe2e1e9e8015e577bdc085c0bb0038081115a7fa13 | FROZEN |
| Difference gradient | SHA256 300dd4f62eb6d13f80e500d5b4769e3ccde6a8a828a0f79444bec0f8933df74a；L2 11.98609163397219 | COMPLETE two-prompt Dev |
| Readouts | Mbpp/14、Mbpp/577 canonical function vs constant-None stub | FROZEN before readout |
| Tested steps | 1e-4、5e-5 | FROZEN |
| Relative residuals | 1.0299787148787611、0.6991867301358323 | FAIL accepted-linearity gate |
| Independent reward consequence | NOT_MEASURED；independent target slopes null | NOT_RUN |
| Decision | 没有 accepted eta；取消当前 C/D 参数更新 | INCONCLUSIVE |

## 2026-09-07 C8/D4 projection diagnostic

| 项目 | 冻结值 | 状态 |
|---|---|---|
| Cycle | 03:50:25–05:50:25 UTC | RUNNING at status snapshot |
| C workload | 8 frozen prompts × 1 original K8 group | PREDECLARED |
| D workload | 4 frozen independent prompts × 1 original K8 group | PREDECLARED |
| Outputs | frozen-function projections + independent-D target projections only | RUNNING / no receipt yet |
| Parameter updates | none, because Dev step gate failed | CANCELED |
| Inference scope | directional diagnostic；not full C32/D32 or complete two-bank inference | FROZEN boundary |

## 后续参数更新前仍须冻结

- 小于 5e-5 的 Dev step 与 half-step；不能根据 C8/D4 投影结果选择。
- accepted-linearity 判据、绝对或相对阈值与比较数量。
- 完整 C32/D32 配置、stage cap、输出目录、预算 ledger 与双银行不确定性方法。
- 独立生成 reward consequence；当前 D target projection 不能替代它。

## 每阶段必须保存

- model/tokenizer revision、本地 checkpoint manifest 与 SHA256、chat-template hash。
- 完整配置、任务 hash、prompt/group/slot/RNG identity、generation config。
- response ids、EOS/截断、sampling/scoring token logp、active mask。
- 每测试 verdict、解析/超时/基础设施状态、test manifest 与 reference receipt。
- S/I/RLOO 权重、全参数 manifest、梯度范围/存储、耗时、峰值显存和生成 token。
- Dev 冻结读出与步长、C/D 独立性、prompt 级统计和四门判决证据。
