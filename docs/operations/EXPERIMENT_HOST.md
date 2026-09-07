# 实验机运行记录

更新时间：2026-09-07。此文件保存现场主机信息；算法和 README 不绑定该机器。

## 连接与工作目录

| 项目 | 当前值 |
|---|---|
| SSH | ssh -p 36203 root@connect.westc.seetacloud.com |
| 仓库 | /root/autodl-tmp/feedback-grpo |
| Python | 3.12 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 5.15.1 |
| NumPy | 2.3.2 |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| 显存 | 32760 MiB 报告值；PyTorch 约 31.48 GiB |
| 执行方式 | 原生 Python；GPU 实验已获用户授权 |

每个 run receipt 记录解释器绝对路径、CUDA、driver、设备名称、包版本与实际模型 manifest。不要覆盖当前 CUDA PyTorch 安装；仅在具体缺包时安装兼容依赖。

## 模型与网络

直接 Hugging Face 路线当前不可用。优先使用 /root/autodl-tmp/models 中已有资产；缺失模型走镜像或 AutoDL 支持路线，并保存精确 revision、本地文件 manifest 与 SHA256。

已保留：

- Qwen2.5-7B-Instruct；当前 GPU-0 revision a09a35458c702b33eeacc393d103063234e8bc28。
- OLMo2-7B 权重，当前未用于 v4 运行。

ModelScope 的 Coder 下载在约 2 GB partial 时已停止，未进入配置或证据。不要把残留当作可用 checkpoint，也无需恢复下载。

## 清理记录

reports/remote_cleanup_20260907.json 是用户授权的旧项目清理回执：

- 移除 307 项旧代码/结果，0 失败。
- 可用空间从 3,141,603,328 bytes 增至 49,742,094,336 bytes，约 2.93 GiB → 46.33 GiB。
- 保留 /root/autodl-tmp/.autodl 与 /root/autodl-tmp/models。

后续不要删除当前模型资产、反馈项目运行目录或未索引的新结果。大型产物写新目录，完成后在本地索引登记 receipt/hash。

## GPU-0 记录

模型合同：Qwen2.5-7B-Instruct、上述 revision、FP32、eager、全参数；1 个 Dev prompt、1 条回答、最多 32 token。

| 尝试 | 现场结果 | 状态 |
|---|---|---|
| v1 | Transformers 5.15 chat template 返回 BatchEncoding；旧 prompt 编码路径在生成前失败 | 已定位并修复统一编码接口 |
| v2 | 生成 32 token；逐 token score tolerance 1e-5 通过；full gradient 申请约 2.03 GiB 时只有约 1.28 GiB 空闲，OOM | PARTIAL |
| v3 | CPU input embedding 与 CPU post-accumulate gradient storage | PASS_PLUMBING |

v3 receipt：reports/gpu0/qwen7b_fp32_cpuemb.json。

| 指标 | 值 |
|---|---:|
| generation_seconds | 2.8519 |
| gradient_seconds | 29.4187 |
| max_token_error | 3.9583e-6 |
| peak_gpu_bytes | 32,080,404,992 |
| parameter scope | all actor parameters |
| natural_coupling_effect | NOT_MEASURED |
| pilot_decision | NOT_RUN |

存储变化不改变全参数范围。该 probe 证明当前 checkpoint 的生成、score point 与全参数梯度 plumbing，不证明自然 S/I 效应。

## Dev 与当前周期

完整 Dev fixed bank 已完成：16 prompts、32 groups、256 responses、24,515 tokens；结果见 `reports/dev_all16_credit_2h.json`。前两题全参数 S−I gradient 已核验，L2 11.98609163397219，远端文件 SHA256 `300dd4f62eb6d13f80e500d5b4769e3ccde6a8a828a0f79444bec0f8933df74a`。

Dev 功能步长 `1e-4` 与 `5e-5` 的 relative residual 为 1.02998 / 0.69919，没有 accepted linear step。因此取消当前 C/D 参数更新。03:50:25–05:50:25 UTC 周期只运行预声明 C8/D4 的固定功能与独立 D 目标投影；投影已完成，详见两小时报告。

Dev 前两题 S-I 的全参数 gradient cache 约 30GB，可由 raw banks 与冻结配置重算。若为 C gradient 释放空间，先复制并核验 receipt、functions、配置和 raw candidate/scored banks，再删除该 cache；另存删除与空间回执。不得删除唯一原始银行。

## 当前主机执行原则

1. C8/D4 投影诊断优先；不并行扫模型、精度、K、mask 或任务。
2. 一次只做与已观测失败对应的兼容/存储修复，并保留上一失败。
3. 每个输出目录只创建一次；失败也保留日志与配置。预算由 scripts/run_pilot.py 的独立 ledger 记账。
4. 代码评分默认原生逐测试 worker。候选超时/语法错误按冻结规则记失败；解释器、资源或 worker 异常暂停银行并单独记录。
5. v4 原 5090/Pro 6000 小时表不适用于当前 4080 SUPER。使用实际 cycle ledger，不承诺旧吞吐。
6. 未通过更小 eta 的 Dev 校准前，不启动参数更新或训练。

## 有效资源限制与首轮实现修正

宿主机 /proc/meminfo 显示约503GiB不能当作实例可用内存。实测 cgroup memory.max=66571993088 bytes（62GiB）。
默认聚合改为一份全局梯度，避免两个约30GB的匿名CPU梯度同时常驻；有需要的逐prompt矩阵路径仍需按所在机器的实际容量规划。
GPU全参数反传使用CPU input embedding、CPU梯度累积及可选激活存储。固定权重与统计平均方式不变。
完整Dev前两题的最终数值容差为2e-4，实际max token error=1.19764e-4；原失败及27候选复用过程见Dev报告。

## 已释放的可重算缓存

04:55:55 UTC 已删除 Dev 前两题 `gradient.safetensors`，30,462,505,048 bytes；原始银行、权重、源码和测量证据保留。回执：`reports/dev_first2/gradient/retention.json`。当前该 Dev 梯度文件已不可直接读取，重跑需重新计算。
