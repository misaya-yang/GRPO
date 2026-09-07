# Dependent feedback v4 运行手册

所有命令从仓库根目录运行。实验机路径与环境快照见 docs/operations/EXPERIMENT_HOST.md。原生 Python 是当前默认执行方式；Docker、chroot 和固定 Conda 路径不是前置条件。

以下变量仅用于缩短示例；实际命令和解析后的绝对路径写入 run receipt：

~~~bash
cd /root/autodl-tmp/feedback-grpo
export PYTHONPATH=src
export RC_PYTHON=python3
export RC_TASKS=data/feedback/mbpp_pilot.jsonl
~~~

直接 Hugging Face 不可用时先用已有本地模型。缺失资产通过镜像或 AutoDL 支持路线获取，并保存 revision 与本地文件 manifest；半下载目录不能进入运行配置。

## 0. 只读核对

~~~bash
git status --short
$RC_PYTHON -V
$RC_PYTHON - <<'PY'
import numpy, torch, transformers
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory)
PY
$RC_PYTHON -m reward_coupling.cli --help
~~~

任务 hash 必须为 a39d757611cb58d9d7f48f49e0920f0be1b4c6dd7a93cd9dcda28bf6f37746b4。运行配置需要 40 位 model/tokenizer revision 和 64 位 task hash。使用 model_path 时，还必须提供 checkpoint_manifest 及其 SHA256。

configs/feedback/pilot_code.json 是模板，仍含旧模型 placeholder，不能直接运行。每次 probe/Dev/C/D 使用新的派生配置，记录实际模型、revision、路径、offload_gradients、token cap 和 stage cap；不覆盖模板或旧配置。

## 1. CPU 前置检查

~~~bash
make test PYTHON=$RC_PYTHON
make exact PYTHON=$RC_PYTHON
~~~

本地已有 64-test 与 exact PASS 回执；实验机重跑只用于环境核对，不提升真实模型证据等级。代码候选评分使用 executor: native，test_python 指向具备测试依赖的解释器。原生 worker 每个测试使用新临时目录与子进程；它是进程隔离。

## 2. GPU-0 单点 probe

当前已通过配置：已有 Qwen2.5-7B-Instruct，revision a09a35458c702b33eeacc393d103063234e8bc28，1 个 Dev 题、1 条回答、32 token，FP32、eager、全参数，CPU input embedding 与 CPU post-accumulate gradient storage。

~~~bash
$RC_PYTHON scripts/run_pilot.py \
  --ledger runs/budget/gpu0.jsonl \
  --cap-seconds GPU0_CAP \
  -- \
  $RC_PYTHON scripts/gpu_probe.py \
    --config runs/configs/gpu0-v3.json \
    --tasks $RC_TASKS \
    --output runs/gpu0-v3
~~~

GPU0_CAP 替换为 ledger 冻结的正整数秒数。输出目录必须不存在。当前回执 reports/gpu0/qwen7b_fp32_cpuemb.json：generation 2.8519 s、gradient 29.4187 s、max token error 3.9583e-6、peak GPU 32,080,404,992 bytes。

诊断历史：

- v1：Transformers 5.15 chat template 返回 BatchEncoding，在生成前失败；统一 prompt 编码接口后重试。
- v2：生成 32 token 且 score tolerance 通过；全参数 FP32 gradient 申请约 2.03 GiB 时仅余约 1.28 GiB，OOM。
- v3：参数范围不变，只改变输入与梯度存储位置，完整 PASS。

这些是方法 plumbing。receipt 内 natural_coupling_effect 与 pilot_decision 仍应是 NOT_MEASURED / NOT_RUN。

## 3. bounded Dev

当前完整 Dev 已完成：16 prompts、32 groups、256 responses、24,515 tokens；见 reports/dev_all16_credit_2h.json。下面命令用于从零复现，不能覆盖现有银行。冻结顺序、每题 2 个 K=8 组和 cap 384 不可根据 prompt 内容或 reward 改变。

collect CLI 按整个 split 收集。完整 Dev 的标准入口为：

~~~bash
$RC_PYTHON scripts/run_pilot.py \
  --ledger runs/budget/dev.jsonl \
  --cap-seconds DEV_CAP \
  -- \
  $RC_PYTHON -m reward_coupling.cli collect \
    --config runs/configs/dev.json \
    --tasks $RC_TASKS \
    --split Dev \
    --output runs/dev-candidates

$RC_PYTHON -m reward_coupling.cli score \
  --bank runs/dev-candidates \
  --tasks $RC_TASKS \
  --output runs/dev-scored

$RC_PYTHON -m reward_coupling.cli integrate \
  --bank runs/dev-scored \
  --output runs/dev-weights
~~~

score 会用冻结 reference program 检查相关题目。基础设施错误暂停银行；语法错误、候选失败和候选超时按合同记零。检查 collection.json、bank receipt、解析/超时分布、每题 verdict 覆盖和实际 token/墙钟。

Dev 上定义少量有符号可微读出。definitions JSON 每行必须是 split: Dev，并包含固定 prompt_ids、response_ids 与 coefficient：

~~~bash
$RC_PYTHON -m reward_coupling.cli freeze-readout \
  --config runs/configs/dev.json \
  --definitions runs/dev-readout-definitions.json \
  --output runs/dev-readout
~~~

查看 C 结果前冻结步长、阈值、比较数量、reference_receipt、dev_freeze 及二者 hash。当前已测 eta=1e-4/5e-5 的 relative residual 为 1.02998/0.69919，均未通过线性门；下一次参数更新前必须在 Dev 预声明小于 5e-5 的 step 与 half-step。随后对 C/D 配置运行 preflight：

~~~bash
$RC_PYTHON -m reward_coupling.cli preflight \
  --config runs/configs/confirmation.json \
  --tasks $RC_TASKS \
  --output runs/preflight-confirmation.json
~~~

状态必须无 blocker，才能收集 C/D。

## 4. C 机制银行

当前两小时周期运行的是预声明 C8/D4 投影诊断：C 8 prompts、D 4 independent prompts、每题 1 个 K8 组，只计算 frozen-function 与 independent-D target projections。它不运行下面的完整 C32/D32 梯度矩阵，也不提供完整推断。

~~~bash
$RC_PYTHON -m reward_coupling.cli collect \
  --config runs/configs/confirmation.json --tasks $RC_TASKS --split C \
  --output runs/C-candidates
$RC_PYTHON -m reward_coupling.cli score \
  --bank runs/C-candidates --tasks $RC_TASKS --output runs/C-scored
$RC_PYTHON -m reward_coupling.cli integrate \
  --bank runs/C-scored --output runs/C-weights

$RC_PYTHON -m reward_coupling.cli audit \
  --bank runs/C-scored --arm difference \
  --output runs/C-difference
~~~

默认先运行一次同银行 difference 全参数审计，保存聚合梯度、逐轨迹权重和 hash receipt；以下投影脚本直接计算冻结函数方向，无需同时保存多个全参数读出。只有实际问题需要且存储预算可容纳时，再运行其他 arm 或逐 prompt 梯度。不要为每个测试重新生成候选，也不要把 integrate 的精确权重当成总体显著性。

## 5. D 独立目标与局部步长

先封存与 C 不重叠的 D 原始银行：

~~~bash
$RC_PYTHON -m reward_coupling.cli collect \
  --config runs/configs/confirmation.json --tasks $RC_TASKS --split D \
  --output runs/D-candidates
$RC_PYTHON -m reward_coupling.cli score \
  --bank runs/D-candidates --tasks $RC_TASKS --output runs/D-scored
~~~

当前已验证的低存储方向诊断路径如下。`functions.json` 必须来自完成的 Dev 校准目录，旁边的 plan/receipt 及函数 hash 会一起校验；C/D 必须使用相同生成配置。此命令只计算方向，不施加参数更新：

~~~bash
$RC_PYTHON scripts/dev_step_calibration.py \
  --bank runs/C-scored --difference runs/C-difference \
  --tasks "$RC_TASKS" \
  --functions runs/Dev-calibration/functions.json \
  --evaluation-bank runs/D-scored \
  --project-evaluation --project-only \
  --output runs/CD-direction
~~~

`predicted_slopes.json` 在 D 开始前落盘；`D_target_rows.partial.jsonl` 每行完成后 flush，属于未封存诊断。只有最终 `receipt.json` 代表全阶段完成。D 目标为原始平均测试 reward 与 suite reward 的方向投影；未测量的零 reward 行有显式占位标记。其点估计不含完整 C/D 置信区间。


使用与 C 不重叠的 D prompt；split 名称是大写 D。以下 branch 命令只有在更小 eta 的 Dev 线性校准通过后才允许运行。当前 C/D 参数更新已取消：

~~~bash
$RC_PYTHON -m reward_coupling.cli branch \
  --shared runs/C-shared \
  --independent runs/C-independent \
  --evaluation-bank runs/D-scored \
  --step FROZEN_DEV_STEP \
  --readout runs/dev-readout \
  --output runs/C-SI-local-step
~~~

branch 验证 C/D prompt、prompt hash、group seed 和 actor RNG 不重叠，并从同一 checkpoint 分别施加共同 eta 与 eta/2。其 likelihood-ratio 结果是局部同银行诊断；receipt 的 independent_generation_after_step 仍为 false。后续独立生成评价需要另存新回执。

统计以 prompt 为单位，分别传播 C/D 不确定性。没有完整双银行矩阵或 interaction 项时，只能报告条件或 first-order 区间。正缩放同时报告 signed scale、non-negative scale 与 residual；只报 cosine 不足以过门。

## 6. 判决与训练门禁

将四门证据及每个文件 SHA256 写入新的 pilot decision JSON，再由 statistics.pilot_decision 得出 GO/STOP/INCONCLUSIVE。准备完成、GPU-0 PASS、配置布尔值或人工修改 pilot_decision 都不能解锁训练。

只有 evidence-bearing GO 后才运行：

~~~bash
$RC_PYTHON -m reward_coupling.cli train \
  --config runs/configs/train-shared.json \
  --tasks $RC_TASKS \
  --decision reports/pilot_decision_TIMESTAMP.json \
  --output runs/train-shared-TIMESTAMP
~~~

训练配置额外冻结 decision_sha256、coupling、advantage、optimizer、updates、learning_rate、prompts_per_update 与 checkpoint_every。training_complete_evaluation_required 仍不代表 held-out 改善；必须用共同 IID 评价。
