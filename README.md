# v5.1 独立分层 GRPO 实验

本项目比较同一 checkpoint、固定 K 和回答预算 N 下，独立分层 Strat-full 与强 IID-all 的梯度估计方差及实际成本。两者以同一个 IID K 更新为目标；分层不是提高未训练模型期望正确率的机制。

当前已完成真实模型小规模实验：4 题、32 个独立宏组、256 回答。方差比 0.9209，耗时修正比 0.9024，但区间宽、结果对题目及单组敏感，结论为 **INCONCLUSIVE**，尚无训练收益证据。见[实验报告](reports/trend_pilot/REPORT.md)、[详细诊断](reports/trend_pilot/ANALYSIS.md)和[当前状态](docs/STATUS.md)。

## 实现与入口

- 独立条件区间采样，使用私有随机流与保留 binary64 输入权重比例的整数 CDF。
- Strat-full、IID-all、Strat-cross 的二值及网格奖励权重；K 保持不变，虚拟组不计作新增样本。
- 共同 LoRA 参数空间、逐 token score 校验、宏组聚合梯度、题目/宏组层级统计与成本记录。
- 原生运行、累计预算、start/status/report 命令，见[执行交接](docs/experiments/V5_1_HANDOFF.md)。
- 共同 checkpoint 与在线训练代码已实现，本轮未运行。

模型是实验载体，不锁定特定型号。优先复用现有权重，记录实际精度、参数空间和 revision。v4 的回答 IID、评分随机性关联实验保留在 `src/reward_coupling/`；更早的回答依赖实现保留在 `src/dependent_rollouts/`，其中采样基础设施由当前主线复用。原始理论档案保持原字节。

## 本地环境

| 机器 | 已知环境 | 本轮角色 |
|---|---|---|
| 当前 Mac | M4 MAX；macOS 26.6.2；Conda `aidemo` | CPU 测试、精确验证和文档；不是固定或唯一开发机 |
| Mac `aidemo` 快照 | Python 3.13.11、NumPy 2.4.2、PyTorch 2.10.0、Transformers 5.2.0、safetensors 0.7.0 | 2026-09-07 查询；后续以现场回执为准 |
| 家用 PC | 已确认存在；OS、CPU/GPU、内存和 Python 环境未知 | 暂不分配固定角色，不猜测配置 |

本轮 Mac 只运行 CPU 工作。算法不绑定机器或 Conda 路径。当前运行器已在 macOS/Linux 验证；原生测试 worker 与预算锁使用 POSIX 接口，家用 PC 的运行方式待环境确认。通过 `PYTHON`、配置中的 `device`、`model_path` 与 `test_python` 记录实际环境。

## 本地检查

```bash
conda activate aidemo
make test
make lint
make exact
PYTHONPATH=src python -m stratified_grpo.cli --help
```

也可以用 `make test PYTHON=/path/to/python` 选择其他已准备环境。当前操作见[执行交接](docs/experiments/V5_1_HANDOFF.md)；旧 v4 操作见 [RUNBOOK](docs/experiments/RUNBOOK.md)。

## 文档职责

- `agent.md`：稳定的项目核心约束。
- `index.md`：代码、理论、运行记录和证据索引。
- `docs/experiments/V5_1_HANDOFF.md`：当前执行入口及已运行范围。
- `docs/experiments/PLAN.md` / `MANIFEST.md`：v4 历史计划与冻结信息。

原始方案、Pro 建议和历史 dossier 保存在 `docs/theory/` 与 `docs/archive/`。不要改写这些原始来源，也不要把 NOT_RUN、准备完成或单点 probe 写成 pilot GO。
